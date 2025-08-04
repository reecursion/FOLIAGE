import os
import torch
import pandas as pd
import numpy as np
import argparse
import re
import gc
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import shutil

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "eager"

# Try to enable flash attention if GPU supports it
if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
    try:
        import flash_attn
        TORCH_DTYPE = torch.bfloat16
        ATTN_IMPLEMENTATION = "flash_attention_2"
        print("Flash Attention 2 enabled")
    except ImportError:
        print("Flash Attention not found, using default attention implementation")

def setup_chat_format(model, tokenizer):
    """Set up chat format for the model if available."""
    try:
        from trl import setup_chat_format
        return setup_chat_format(model, tokenizer)
    except ImportError:
        print("Warning: setup_chat_format not available, skipping")
        return model, tokenizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Finetune LLM with PEFT and K-Fold Cross Validation")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="cb", choices=["cb", "p4g", "custom"])
    
    # K-fold arguments
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--run_fold", type=int, default=None)
    
    # Intentions/scaffolding arguments 
    parser.add_argument("--scaffolding_type", type=str, default="none", 
                        choices=["local", "global", "both", "none"])
    parser.add_argument("--summary_type", type=str, default="none",
                        choices=["none", "traditional", "scd", "relational", "scm", 
                                 "politeness_theory_stage2"])
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/data/user_data/gganeshl/llama-finetuned")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)  
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=3)  
    
    # Evaluation arguments
    parser.add_argument("--eval_metric", type=str, default=None, 
                      choices=["nmse", "rmse", "pearson", "accuracy", "loss"],
                      help="Metric to use for model evaluation (default: nmse for cb, accuracy for p4g, loss otherwise)")
    parser.add_argument("--compute_metrics", action="store_true", 
                      help="Use compute_metrics function for evaluation instead of loss")
    
    # Hugging Face arguments
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    
    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="LLM-Finetuning")
    
    # System instruction
    parser.add_argument("--system_instruction", type=str, 
                        default="You are a helpful assistant that provides accurate and concise information.")
    
    # Memory optimization
    parser.add_argument("--memory_efficient", action="store_true", 
                      help="Enable memory optimizations to reduce VRAM usage")
    
    # Evaluation batch size
    parser.add_argument("--eval_batch_size", type=int, default=1,
                      help="Batch size for evaluation")
    
    return parser.parse_args()

def setup_tracking(args, fold=None):
    """Setup tracking with WandB and Hugging Face."""
    if args.use_wandb:
        # Create run name with config details
        run_name = f"{args.dataset_type}_{args.scaffolding_type}"
        
        # Add summary type if applicable
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            run_name += f"_{args.summary_type}"
        
        # Extract ratio from the dataset path if available
        ratio_match = re.search(r'ratio_(\d+\.\d+|\d+)', args.dataset_path)
        if ratio_match:
            run_name += f"_ratio_{ratio_match.group(1)}"
            
        # Add fold information
        if fold is not None:
            run_name += f"_fold_{fold}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            reinit=True,
            config={
                "base_model": args.base_model,
                "dataset_type": args.dataset_type,
                "scaffolding_type": args.scaffolding_type,
                "summary_type": args.summary_type,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "lora_rank": args.lora_rank,
                "early_stopping_patience": args.early_stopping_patience,
                "fold": fold,
                "eval_metric": args.eval_metric,
                "compute_metrics": args.compute_metrics
            }
        )
    
    if args.push_to_hub:
        # First try environment variable, fall back to argument
        hf_token = os.environ.get("HF_API_KEY", args.hf_token)
        if hf_token:
            login(token=hf_token)
        else:
            print("No Hugging Face token found. Set HF_API_KEY environment variable or use --hf_token")
            args.push_to_hub = False

def prepare_dataset(args, tokenizer, fold_indices=None):
    """Prepare dataset for fine-tuning."""
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    # Check if dataset is in utterance level format
    if not all(col in df.columns for col in ['dialogue_id', 'utterance_idx', 'speaker', 'utterance']):
        print("Dataset not in utterance-level format. Not supported.")
        return None
        
    # Group by dialogue_id
    grouped_data = []
    
    for dialogue_id, group in df.groupby('dialogue_id'):
        # Sort by utterance index
        group = group.sort_values('utterance_idx')
        
        # Format conversation
        conversation_lines = []
        for _, row in group.iterrows():
            utterance = row['utterance']
            # Add intention if using local scaffolding
            if args.scaffolding_type in ["local", "both"] and 'intention' in row and pd.notna(row['intention']):
                conversation_lines.append(f"{row['speaker']}: {utterance} [{row['intention']}]")
            else:
                conversation_lines.append(f"{row['speaker']}: {utterance}")
        
        formatted_conversation = "\n".join(conversation_lines)
        
        # Add summary if using global scaffolding
        summary = ""
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            summary_column = f"{args.summary_type}_summary"
            if summary_column in group.columns and pd.notna(group[summary_column].iloc[0]):
                summary = f", [Summary: {group[summary_column].iloc[0]}]"
        
        # Format based on dataset type
        actual_value = None
        buyer_target = seller_target = None
        
        if args.dataset_type == "cb":  # Car Buying dataset
            actual_value = group['sale_price'].iloc[0]
            outcome = f"FINAL_PRICE: ${actual_value}"
            buyer_target = group['buyer_target'].iloc[0]
            seller_target = group['seller_target'].iloc[0]
            
            # Format prompt
            intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
            summary_part = f", [summary]" if args.scaffolding_type in ["global", "both"] and args.summary_type != "none" else ""
            
            input_text = f"<s>[INST] Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: $[number]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}> [/INST]"
            
        elif args.dataset_type == "p4g":  # Persuasion for Good dataset
            donation_amount = group.get('donation_amount', None)
            if donation_amount is not None and not pd.isna(donation_amount.iloc[0]):
                donation_value = donation_amount.iloc[0]
                outcome = f"DONATION: YES, AMOUNT: ${donation_value}"
                actual_value = 1
            else:
                outcome = "DONATION: YES" if group['label'].iloc[0].lower() == 'yes' else "DONATION: NO"
                actual_value = 1 if group['label'].iloc[0].lower() == 'yes' else 0
            
            input_text = f"<s>[INST] You are helping analyze a persuasion conversation.\n\nConversation:\n{formatted_conversation}{summary}\n\nPredict whether the persuadee will make a donation. [/INST]"
        
        else:  # Custom dataset
            input_text = f"<s>[INST] {args.system_instruction}\n\nConversation:\n{formatted_conversation}{summary}\n\nPlease respond appropriately. [/INST]"
            
            if 'response' in group.columns:
                outcome = group['response'].iloc[0]
            else:
                outcome_col = next((col for col in ['output', 'target', 'label'] if col in group.columns), None)
                outcome = group[outcome_col].iloc[0] if outcome_col else "No response available."
        
        # Format completion
        output_text = f" {outcome} </s>"
        
        # Create dataset entry
        entry = {
            "dialogue_id": dialogue_id,
            "input": input_text,
            "output": output_text,
            "text": input_text + output_text,
            "actual_value": actual_value
        }
        
        # Add buyer and seller targets for CB dataset
        if args.dataset_type == "cb":
            entry["buyer_target"] = buyer_target
            entry["seller_target"] = seller_target
        
        grouped_data.append(entry)
    
    # Create dataset dictionary
    dataset_dict = {key: [item.get(key) for item in grouped_data] for key in 
                    ["dialogue_id", "input", "output", "text", "actual_value"]}
    
    # Add buyer and seller targets for CB dataset
    if args.dataset_type == "cb":
        dataset_dict["buyer_target"] = [item.get("buyer_target") for item in grouped_data]
        dataset_dict["seller_target"] = [item.get("seller_target") for item in grouped_data]
    
    # Create dataset
    full_dataset = Dataset.from_dict(dataset_dict)
    
    # Apply k-fold split if fold_indices is provided
    if fold_indices is not None:
        train_dataset = full_dataset.select(fold_indices['train'])
        eval_dataset = full_dataset.select(fold_indices['eval'])
        return {"train": train_dataset, "test": eval_dataset}
    else:
        return full_dataset

def setup_model_and_tokenizer(args, only_token=False):
    """Set up the model with quantization and PEFT."""
    # Clean up GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Enable memory optimizations if requested
    if args.memory_efficient or True:  # Always enable for safety
        # Use expandable segments to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # BitsAndBytes config for 4-bit quantization 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=TORCH_DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set pad token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    # Set sequence length
    if hasattr(tokenizer, 'model_max_length') and args.max_length > 0:
        tokenizer.model_max_length = args.max_length

    if only_token:
        return None, tokenizer 
    # Load model with memory optimizations
    print(f"Loading model from {args.base_model}")
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": TORCH_DTYPE,
    }
    
    if args.memory_efficient:
        # Add memory-efficient options
        model_kwargs.update({
            "max_memory": {0: "32GB"},  # Limit memory usage
            "offload_folder": "offload_folder",
            "offload_state_dict": True
        })
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    
    # Find target modules for LoRA
    def find_target_modules(model):
        import bitsandbytes as bnb
        modules = set()
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                names = name.split('.')
                modules.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in modules:
            modules.remove('lm_head')
        return list(modules)
    
    target_modules = find_target_modules(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    # Apply PEFT
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def extract_price(text):
    """Extract price from text in the format 'FINAL_PRICE: $XXX'."""
    price_pattern = r'FINAL_PRICE: \$(\d+(\.\d+)?)'
    match = re.search(price_pattern, text)
    if match:
        return float(match.group(1))
    return None

def extract_donation(text):
    """Extract donation information from text."""
    if "DONATION: YES" in text:
        # Try to extract amount if present
        amount_match = re.search(r'AMOUNT: \$(\d+(\.\d+)?)', text)
        if amount_match:
            return 1, float(amount_match.group(1))
        return 1, None
    elif "DONATION: NO" in text:
        return 0, None
    return None, None

# Define a function to extract actual values from the evaluation dataset
def get_actual_values(dataset, dataset_type):
    """Extract actual values from the dataset."""
    if dataset_type == "cb":
        return dataset["sale_price"]
    elif dataset_type == "p4g":
        return dataset["actual_value"]
    return None

# Define a more memory-efficient compute_metrics function that processes one example at a time
def compute_metrics(eval_preds, args, tokenizer, eval_dataset):
    """Compute evaluation metrics without using batch_decode."""
    preds_ids, _ = eval_preds
    
    # Get actual values directly from the dataset
    actual_values = get_actual_values(eval_dataset, args.dataset_type)
    
    # Initialize predicted values list
    predicted_values = []
    
    # Process each prediction one at a time to avoid memory issues
    for i in range(len(preds_ids)):
        # Decode a single prediction at a time
        decoded_pred = tokenizer.decode(preds_ids[i], skip_special_tokens=True)
        
        # Extract the relevant value based on dataset type
        if args.dataset_type == "cb":
            predicted_value = extract_price(decoded_pred)
        elif args.dataset_type == "p4g":
            predicted_value, _ = extract_donation(decoded_pred)
        else:
            predicted_value = None
            
        predicted_values.append(predicted_value)
        
        # Free memory
        del decoded_pred
        
        # Garbage collect periodically
        if i % 10 == 0 and i > 0:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    
    # Calculate metrics based on dataset type
    metrics = {}
    
    if args.dataset_type == "cb":
        # Filter out None values
        valid_pairs = [(p, a) for p, a in zip(predicted_values, actual_values) 
                     if p is not None and a is not None and a != 0]
        
        if valid_pairs:
            # Unpack the pairs
            preds, actuals = zip(*valid_pairs)
            
            # Calculate NMSE
            nmse_values = [(a - p) ** 2 / a for p, a in valid_pairs]
            metrics['nmse'] = np.mean(nmse_values)
            
            # Calculate RMSE
            mse = mean_squared_error(actuals, preds)
            metrics['rmse'] = np.sqrt(mse)
            
            # Calculate Pearson correlation
            if len(valid_pairs) > 1:
                pearson_corr, _ = pearsonr(actuals, preds)
                metrics['pearson'] = pearson_corr
    
    elif args.dataset_type == "p4g":
        # Filter out None values
        valid_pairs = [(p, a) for p, a in zip(predicted_values, actual_values) 
                     if p is not None and a is not None]
        
        if valid_pairs:
            # Calculate accuracy
            correct = sum(1 for p, a in valid_pairs if p == a)
            metrics['accuracy'] = correct / len(valid_pairs)
    
    # Print some example predictions for debugging (limited number)
    print("\nExample predictions (first 3):")
    for i in range(min(3, len(predicted_values))):
        # Decode a single example for display
        sample_pred = tokenizer.decode(preds_ids[i], skip_special_tokens=True)
        print(f"Prediction: {sample_pred[:100]}...")  # Print only first 100 chars
        print(f"Extracted value: {predicted_values[i]}")
        print(f"Actual value: {actual_values[i] if i < len(actual_values) else None}")
        print("---")
        del sample_pred  # Clean up
    
    # Clean up
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return metrics

def create_predictions_df(test_dataset, model, tokenizer, dataset_type, fold=None):
    """Generate predictions and create DataFrame with results."""
    model.eval()
    all_results = []
    predicted_values = []
    actual_values = []
    
    # Use smaller batch for prediction to save memory
    PREDICT_BATCH_SIZE = 1
    
    # Process in batches
    for i in range(0, len(test_dataset), PREDICT_BATCH_SIZE):
        # Get batch
        batch = test_dataset.select(range(i, min(i + PREDICT_BATCH_SIZE, len(test_dataset))))
        
        if i > 0 and i % 10 == 0:
            print(f"Processed {i}/{len(test_dataset)} examples")
            # Clear cache periodically
            torch.cuda.empty_cache()
            gc.collect()
        
        for j, example in enumerate(batch):
            # Tokenize input
            inputs = tokenizer(example['input'], return_tensors='pt').to(model.device)
            
            # Generate with specific parameters
            gen_kwargs = {
                'max_new_tokens': 20,
                'temperature': 0.1,
                'do_sample': False,
                'top_p': 0.9,
                'top_k': 0,
                'bos_token_id': tokenizer.bos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token_id': tokenizer.pad_token_id
            }
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask, 
                    **gen_kwargs
                )
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create result entry
            result = {
                'dialogue_id': example['dialogue_id'],
                'prompt': example['input'],
                'actual_answer': example['output'],
                'predicted_answer': output_text,
                'fold': fold
            }
            
            # Extract values based on dataset type
            if dataset_type == "cb":
                pred_price = extract_price(output_text)
                actual_price = example['actual_value']
                result['predicted_final_price'] = pred_price
                result['sale_price'] = actual_price
                
                predicted_values.append(pred_price)
                actual_values.append(actual_price)
                
                # Add buyer and seller targets and calculate success score
                if 'buyer_target' in example and 'seller_target' in example:
                    buyer_target = example['buyer_target']
                    seller_target = example['seller_target']
                    result['buyer_target'] = buyer_target
                    result['seller_target'] = seller_target
                    
                    # Calculate success score if all values are available
                    if pred_price is not None and seller_target != buyer_target:
                        result['r_sale'] = (actual_price - buyer_target) / (seller_target - buyer_target)
                        result['r_predicted'] = (pred_price - buyer_target) / (seller_target - buyer_target)
            
            elif dataset_type == "p4g":
                pred_donation, pred_amount = extract_donation(output_text)
                result['predicted_donation'] = pred_donation
                if pred_amount is not None:
                    result['predicted_amount'] = pred_amount
                result['actual_donation'] = example['actual_value']
                
                predicted_values.append(pred_donation)
                actual_values.append(example['actual_value'])
            
            all_results.append(result)
            
            # Free memory
            del inputs, outputs
            
    # Create dataframe from results
    predictions_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    return predictions_df, predicted_values, actual_values

def train_model(args, model, tokenizer, dataset, fold=None):
    """Train the model with early stopping and evaluate it."""
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    output_dir = os.path.join(args.output_dir, f"{args.dataset_type}{fold_suffix}")
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.001 
    )
    
    # Determine metric for evaluation
    if args.eval_metric is None:
        # Choose default metric based on dataset type
        if args.dataset_type == "cb":
            metric_for_best_model = "eval_nmse" if args.compute_metrics else "eval_loss"
            print(f"Metric used: {metric_for_best_model}")
            greater_is_better = False
        elif args.dataset_type == "p4g":
            metric_for_best_model = "eval_accuracy" if args.compute_metrics else "eval_loss"
            greater_is_better = True
        else:
            metric_for_best_model = "eval_loss"
            greater_is_better = False
    else:
        # Use user-specified metric
        metric_for_best_model = f"eval_{args.eval_metric}"
        # Determine if higher is better based on metric
        greater_is_better = args.eval_metric in ["accuracy", "pearson"]
    
    # Set evaluation batch size (smaller than training batch size)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else max(1, args.batch_size // 2)
    
    # Create SFT training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,  # Use smaller batch for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=args.logging_steps,
        eval_strategy="steps",  # Always evaluate at steps
        eval_steps=0.1,  # Evaluate every 10% of training
        save_strategy="steps",
        save_steps=0.2,  # Save every 20% of training
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        max_seq_length=args.max_length,
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    
    # Add memory optimizations
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    # Create compute_metrics function if enabled
    if args.compute_metrics:
        def compute_metrics_wrapper(eval_preds):
            return compute_metrics(eval_preds, args, tokenizer, dataset["test"])
        metrics_fn = compute_metrics_wrapper
    else:
        metrics_fn = None
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=None,  # PEFT is already applied to the model
        args=training_args,
        callbacks=[early_stopping_callback],
        compute_metrics=metrics_fn if args.compute_metrics else None,
        processing_class=tokenizer 
    )
    
    # Train the model
    print(f"Starting training for fold {fold}")
    trainer.train()
    print(f"Training completed for fold {fold}")
    
    # Save the best model
    best_model_path = os.path.join(best_model_dir, "final")
    os.makedirs(best_model_path, exist_ok=True)
    trainer.save_model(best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Clean up memory before generating predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    # Generate predictions and evaluate
    print("Generating predictions...")
    predictions_df, predicted_values, actual_values = create_predictions_df(
        dataset["test"], model, tokenizer, args.dataset_type, fold
    )
    
    # Compute metrics
    metrics = {}
    if args.dataset_type == "cb":
        # Filter out None values
        valid_pairs = [(p, a) for p, a in zip(predicted_values, actual_values) 
                      if p is not None and a is not None and a != 0]
        
        if valid_pairs:
            # Unpack the pairs
            preds, actuals = zip(*valid_pairs)
            
            # Calculate NMSE
            nmse_values = [(a - p) ** 2 / a for p, a in valid_pairs]
            metrics['nmse'] = np.mean(nmse_values)
            
            # Calculate RMSE
            mse = mean_squared_error(actuals, preds)
            metrics['rmse'] = np.sqrt(mse)
            
            # Calculate Pearson correlation
            if len(valid_pairs) > 1:
                pearson_corr, _ = pearsonr(actuals, preds)
                metrics['pearson'] = pearson_corr
    
    elif args.dataset_type == "p4g":
        # Filter out None values
        valid_pairs = [(p, a) for p, a in zip(predicted_values, actual_values) 
                      if p is not None and a is not None]
        
        if valid_pairs:
            # Calculate accuracy
            correct = sum(1 for p, a in valid_pairs if p == a)
            metrics['accuracy'] = correct / len(valid_pairs)
    
    print(f"Evaluation metrics: {metrics}")
    
    return best_model_path, metrics, predictions_df

def save_predictions(predictions_df, args, fold=None):
    """Save predictions to a CSV file with a consistent naming convention."""
    # Create output directory
    output_dir = os.path.join(args.output_dir, "results", args.dataset_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine scaffolding suffix
    scaffolding_map = {"local": "local", "global": "global", "both": "dual", "none": "none"}
    scaffolding_suffix = scaffolding_map[args.scaffolding_type]
    
    # Extract ratio from dataset path
    ratio_match = re.search(r'ratio_(\d+\.\d+|\d+)', args.dataset_path)
    ratio = ratio_match.group(1) if ratio_match else "100"
    
    # Determine summary suffix
    summary_suffix = ""
    if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
        summary_suffix = f"_{args.summary_type}"
    
    # Create output filename
    output_file = os.path.join(
        output_dir, 
        f"predictions_{scaffolding_suffix}_ratio_{ratio}{summary_suffix}.csv"
    )
    
    # Save predictions
    if os.path.exists(output_file):
        # Update existing file
        existing_df = pd.read_csv(output_file)
        
        # Remove existing predictions for this fold
        if fold is not None and 'fold' in predictions_df.columns:
            existing_df = existing_df[existing_df['fold'] != fold]
            
        # Combine dataframes
        combined_df = pd.concat([existing_df, predictions_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Updated predictions in {output_file}")
    else:
        # Create new file
        predictions_df.to_csv(output_file, index=False)
        print(f"Created new predictions file at {output_file}")
    
    return output_file

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print(f"Fine-tuning with PEFT and {args.k_folds}-fold Cross Validation")
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Scaffolding type: {args.scaffolding_type}")
    print(f"Memory efficient mode: {args.memory_efficient}")
    print(f"Compute metrics: {args.compute_metrics}")
    if args.eval_metric:
        print(f"Evaluation metric: {args.eval_metric}")
    
    # Load tokenizer for consistent preprocessing
    _, tokenizer = setup_model_and_tokenizer(args, True)
    
    # Load dataset
    full_dataset = prepare_dataset(args, tokenizer)
    
    # Setup K-fold cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(full_dataset))
    fold_indices = []
    for train_idx, test_idx in kf.split(all_indices):
        fold_indices.append({
            'train': train_idx.tolist(),
            'eval': test_idx.tolist()
        })
    
    # Determine folds to run
    folds_to_run = [args.run_fold] if args.run_fold is not None else range(args.k_folds)
    all_metrics = []
    all_prediction_files = []
    
    # Track best model across all folds
    best_fold = None
    best_metric = float('inf')
    best_model_path = None
    
    # Determine metric for tracking best model
    if args.dataset_type == "cb":
        metric_key = "nmse"
        is_higher_better = False
    elif args.dataset_type == "p4g":
        metric_key = "accuracy"
        is_higher_better = True
    else:
        # Default for custom datasets
        metric_key = args.eval_metric if args.eval_metric else "nmse"
        is_higher_better = metric_key in ["accuracy", "pearson"]
    
    # Run training for each fold
    for fold_idx in folds_to_run:
        if fold_idx >= args.k_folds:
            print(f"Fold {fold_idx} is out of range (max is {args.k_folds-1}). Skipping.")
            continue
            
        print(f"\n{'='*50}")
        print(f"Starting fold {fold_idx + 1}/{args.k_folds}")
        print(f"{'='*50}")
        
        # Setup tracking
        setup_tracking(args, fold=fold_idx)
        
        # Get dataset split for this fold
        fold_dataset = prepare_dataset(args, tokenizer, fold_indices=fold_indices[fold_idx])
        
        # Setup model
        model, tokenizer = setup_model_and_tokenizer(args)
        
        # Apply chat format
        model, tokenizer = setup_chat_format(model, tokenizer)
        
        # Train model and get predictions
        model_path, metrics, predictions_df = train_model(args, model, tokenizer, fold_dataset, fold=fold_idx)
        
        # Store metrics
        if metrics:
            metrics['fold'] = fold_idx
            all_metrics.append(metrics)
            
            # Check if this fold has the best metric
            if metric_key in metrics:
                current_metric = metrics[metric_key]
                is_better = (not is_higher_better and current_metric < best_metric) or \
                            (is_higher_better and current_metric > best_metric)
                
                if is_better:
                    best_metric = current_metric
                    best_fold = fold_idx
                    best_model_path = model_path
                    print(f"New best model from fold {fold_idx} with {metric_key}: {best_metric:.6f}")
        
        # Save predictions
        if predictions_df is not None and not predictions_df.empty:
            prediction_file = save_predictions(predictions_df, args, fold=fold_idx)
            all_prediction_files.append(prediction_file)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Finish WandB run
        if args.use_wandb and wandb.run is not None:
            wandb.finish()
    
    # Report metrics
    if all_metrics:
        print("\nCross-validation complete. Average metrics:")
        
        # Calculate averages for all metrics
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())
        metric_keys.discard('fold')
        
        # Print average metrics
        for key in sorted(metric_keys):
            values = [m.get(key) for m in all_metrics if key in m and m.get(key) is not None]
            if values:
                avg = sum(values) / len(values)
                print(f"Average {key}: {avg:.4f}")
        
        # Print metrics by fold
        print("\nMetrics by fold:")
        for metrics in all_metrics:
            if 'fold' in metrics:
                fold = metrics['fold']
                print(f"Fold {fold}:")
                for key in sorted(metric_keys):
                    if key in metrics:
                        print(f"  {key} = {metrics[key]:.4f}")
        
        # Copy best model to final location
        if best_fold is not None:
            print(f"\nBest model from fold {best_fold}")
            print(f"Best model path: {best_model_path}")
            
            # Extract ratio for filename
            ratio_match = re.search(r'ratio_(\d+\.\d+|\d+)', args.dataset_path)
            ratio = ratio_match.group(1) if ratio_match else "100"
            
            # Create final best model path
            best_model_final_path = os.path.join(
                args.output_dir, 
                f"{args.dataset_type}_{args.scaffolding_type}_ratio_{ratio}_{args.summary_type}_best"
            )
            
            if os.path.exists(best_model_final_path):
                shutil.rmtree(best_model_final_path)
            
            shutil.copytree(best_model_path, best_model_final_path)
            print(f"Best model copied to {best_model_final_path}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()