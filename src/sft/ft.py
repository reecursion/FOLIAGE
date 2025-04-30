import os
import torch
import pandas as pd
import numpy as np
import argparse
import re
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# All imports at the top
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
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

# Detect and configure CUDA/flash-attention at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "eager"

if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
    try:
        import flash_attn
        TORCH_DTYPE = torch.bfloat16
        ATTN_IMPLEMENTATION = "flash_attention_2"
        print("Flash Attention 2 enabled")
    except ImportError:
        print("Flash Attention not found, using default attention implementation")

# Setup chat format helper
def setup_chat_format(model, tokenizer):
    try:
        from trl import setup_chat_format
        return setup_chat_format(model, tokenizer)
    except ImportError:
        print("Warning: setup_chat_format not available, skipping")
        return model, tokenizer

def parse_arguments():
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
                                "appraisal_theory", "politeness_theory_stage2"])
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/data/user_data/gganeshl/llama-finetuned")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=10)
    
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
    
    # CSV output for predictions
    parser.add_argument("--predictions_output", type=str, default="predictions.csv")
    
    return parser.parse_args()

def setup_tracking(args, fold=None):
    """Setup tracking with WandB and Hugging Face."""
    if args.use_wandb:
        try:
            wandb.login()
            run_name = f"fold-{fold}" if fold is not None else None
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                reinit=True
            )
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            args.use_wandb = False
    
    if args.push_to_hub:
        try:
            # First try environment variable, fall back to argument
            hf_token = os.environ.get("HF_API_KEY", args.hf_token)
            if hf_token:
                login(token=hf_token)
            else:
                print("No Hugging Face token found. Set HF_API_KEY environment variable or use --hf_token")
                args.push_to_hub = False
        except Exception as e:
            print(f"Error logging in to Hugging Face Hub: {e}")
            args.push_to_hub = False

def prepare_dataset(args, tokenizer, fold_indices=None):
    """Prepare dataset for fine-tuning."""
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    # Check if dataset is in utterance level format
    is_utterance_level = all(col in df.columns for col in ['dialogue_id', 'utterance_idx', 'speaker', 'utterance'])
    if not is_utterance_level:
        print("Dataset not in utterance-level format. Not supported.")
        return None
        
    # Group by dialogue_id
    grouped_data = []
    
    for dialogue_id, group in df.groupby('dialogue_id'):
        # Sort by utterance index
        group = group.sort_values('utterance_idx')
        
        # Format conversation
        conversation = []
        for _, row in group.iterrows():
            utterance = row['utterance']
            if args.scaffolding_type in ["local", "both"] and 'intention' in row and pd.notna(row['intention']):
                conversation.append(f"{row['speaker']}: {utterance} [{row['intention']}]")
            else:
                conversation.append(f"{row['speaker']}: {utterance}")
        
        formatted_conversation = "\n".join(conversation)
        
        # Add summary if using global scaffolding
        summary = ""
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            summary_column = f"{args.summary_type}_summary"
            if summary_column in group.columns and pd.notna(group[summary_column].iloc[0]):
                summary = f", [Summary: {group[summary_column].iloc[0]}]"
        
        # Format based on dataset type
        actual_value = None
        
        if args.dataset_type == "cb":
            actual_value = group['sale_price'].iloc[0]
            outcome = f"FINAL_PRICE: ${actual_value}"
            buyer_target = group['buyer_target'].iloc[0]
            seller_target = group['seller_target'].iloc[0]
            
            intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
            summary_part = f", [summary]" if args.scaffolding_type in ["global", "both"] and args.summary_type != "none" else ""
            
            input_text = f"<s>[INST] Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: $[number]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}> [/INST]"
            
        elif args.dataset_type == "p4g":
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
        
        grouped_data.append({
            "dialogue_id": dialogue_id,
            "input": input_text,
            "output": output_text,
            "text": input_text + output_text,
            "actual_value": actual_value,
            "buyer_target": buyer_target if args.dataset_type == "cb" else None,
            "seller_target": seller_target if args.dataset_type == "cb" else None
        })
    
    dataset_dict = {
        "dialogue_id": [item["dialogue_id"] for item in grouped_data],
        "input": [item["input"] for item in grouped_data],
        "output": [item["output"] for item in grouped_data],
        "text": [item["text"] for item in grouped_data],
        "actual_value": [item["actual_value"] for item in grouped_data]
    }
    
    # Add buyer and seller targets for CB dataset
    if args.dataset_type == "cb":
        dataset_dict["buyer_target"] = [item["buyer_target"] for item in grouped_data]
        dataset_dict["seller_target"] = [item["seller_target"] for item in grouped_data]
    
    # Create dataset
    full_dataset = Dataset.from_dict(dataset_dict)
    
    # Apply k-fold split if fold_indices is provided
    if fold_indices is not None:
        train_dataset = full_dataset.select(fold_indices['train'])
        eval_dataset = full_dataset.select(fold_indices['eval'])
        return {"train": train_dataset, "test": eval_dataset}
    else:
        return full_dataset

def setup_model_and_tokenizer(args):
    """Set up the model with quantization and PEFT."""
    # Clean up GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
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

    # Load model
    print(f"Loading model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
        offload_folder="offload",   
    )
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

def evaluate_model(model, dataset, tokenizer, fold=None, args=None):
    """Evaluate the model and calculate metrics including NMSE and Success Score RMSE."""
    if 'actual_value' not in dataset.column_names:
        return None, []
    
    price_pattern = r'FINAL_PRICE: \$(\d+(\.\d+)?)'
    predictions = []
    actual_values = []
    dialogue_ids = []
    buyer_targets = []
    seller_targets = []
    
    print(f"Evaluating model on {len(dataset)} examples")
    for item in dataset:
        try:
            inputs = tokenizer(item['input'], return_tensors='pt').to(model.device)

            gen_kwargs = {
                'max_new_tokens': 20,
                'temperature': 0.1,
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
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            match = re.search(price_pattern, output_text)
            if match:
                predicted_price = float(match.group(1))
                predictions.append(predicted_price)
                actual_values.append(item['actual_value'])
                dialogue_ids.append(item['dialogue_id'])
                
                if 'buyer_target' in item and 'seller_target' in item:
                    buyer_targets.append(item.get('buyer_target'))
                    seller_targets.append(item.get('seller_target'))
                else:
                    buyer_targets.append(None)
                    seller_targets.append(None)
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    if not predictions:
        return None, []
    
    # Create a dataframe to store predictions
    results_df = pd.DataFrame({
        'dialogue_id': dialogue_ids,
        'predicted_final_price': predictions,
        'sale_price': actual_values,
        'fold': [fold] * len(predictions),
    })
    
    # Add buyer and seller targets if available
    if all(bt is not None for bt in buyer_targets) and all(st is not None for st in seller_targets):
        results_df['buyer_target'] = buyer_targets
        results_df['seller_target'] = seller_targets
        
        # Calculate success score (r_sale and r_predicted)
        results_df['r_sale'] = (results_df['sale_price'] - results_df['buyer_target']) / (results_df['seller_target'] - results_df['buyer_target'])
        results_df['r_predicted'] = (results_df['predicted_final_price'] - results_df['buyer_target']) / (results_df['seller_target'] - results_df['buyer_target'])
    
    # Calculate metrics
    metrics = {}
    metrics['mse'] = mean_squared_error(actual_values, predictions)
    
    # Calculate RMSE
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Calculate NMSE (Normalized Mean Squared Error)
    normalized_squared_errors = [(actual - pred) ** 2 / actual for actual, pred in zip(actual_values, predictions)]
    metrics['nmse'] = np.mean(normalized_squared_errors)
    
    # Calculate Success Score RMSE if the data is available
    if 'r_sale' in results_df.columns and 'r_predicted' in results_df.columns:
        metrics['r_mse'] = mean_squared_error(results_df['r_sale'], results_df['r_predicted'])
        metrics['r_rmse'] = np.sqrt(metrics['r_mse'])
        
        # Calculate Pearson correlation for success score
        pearson_corr, _ = pearsonr(results_df['r_sale'], results_df['r_predicted'])
        metrics['r_pearson'] = pearson_corr
    
    # Log to wandb if available
    if args.use_wandb and wandb.run is not None:
        for metric_name, metric_value in metrics.items():
            wandb.log({f"eval/{metric_name}": metric_value})
    
    return metrics, results_df

def train_model(args, model, tokenizer, dataset, fold=None):
    """Train the model."""
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    output_dir = os.path.join(args.output_dir, f"{args.dataset_type}{fold_suffix}")
    
    # Create SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        max_seq_length=args.max_length
    )
    
    # Initialize trainer
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=None,  # PEFT is already applied to the model
            processing_class=tokenizer,
            args=training_args
        )
        
        # Train
        print(f"Starting training for fold {fold}")
        trainer_stats = trainer.train()
        print(f"Training completed for fold {fold}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None, {}, None
    
    # Evaluate model and get metrics and predictions
    metrics, predictions_df = evaluate_model(model, dataset["test"], tokenizer, fold, args)
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Get metrics
    if hasattr(trainer_stats, 'metrics'):
        train_metrics = trainer_stats.metrics
        if metrics:
            train_metrics.update(metrics)
        metrics = train_metrics
    
    return output_dir, metrics, predictions_df

def update_predictions_file(predictions_df, args):
    """Update the predictions file with new results."""
    output_file = os.path.join(args.output_dir, args.predictions_output)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If file already exists, append to it
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        # Remove existing predictions for the same fold to avoid duplicates
        if 'fold' in predictions_df.columns:
            fold = predictions_df['fold'].iloc[0]
            existing_df = existing_df[existing_df['fold'] != fold]
        combined_df = pd.concat([existing_df, predictions_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Updated predictions in {output_file}")
    else:
        # Create new file
        predictions_df.to_csv(output_file, index=False)
        print(f"Created new predictions file at {output_file}")
    
    return output_file

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Fine-tuning with PEFT and {args.k_folds}-fold Cross Validation")
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {DEVICE}, Attention implementation: {ATTN_IMPLEMENTATION}")
    
    # Load tokenizer for consistent preprocessing
    _, tokenizer = setup_model_and_tokenizer(args)
    
    # Load dataset
    full_dataset = prepare_dataset(args, tokenizer)
    
    # Setup K-fold 
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
        model, _ = setup_model_and_tokenizer(args)
        
        # Apply chat format
        try:
            model, tokenizer = setup_chat_format(model, tokenizer)
        except Exception as e:
            print(f"Warning: Could not apply chat format: {e}")
        
        model.resize_token_embeddings(len(tokenizer))
        
        # Train model and get predictions
        _, metrics, predictions_df = train_model(args, model, tokenizer, fold_dataset, fold=fold_idx)
        
        # Store metrics
        if metrics:
            metrics['fold'] = fold_idx
            all_metrics.append(metrics)
        
        # Update predictions file after each fold
        if predictions_df is not None and not predictions_df.empty:
            update_predictions_file(predictions_df, args)
        
        # Finish WandB run
        if args.use_wandb and wandb.run is not None:
            wandb.finish()
    
    # Report metrics
    if all_metrics:
        print("\nCross-validation complete. Average metrics:")
        
        # Calculate averages for metrics
        for key in ['train_loss', 'eval_loss', 'mse', 'rmse', 'nmse', 'r_rmse', 'r_pearson']:
            values = [m.get(key) for m in all_metrics if key in m and m.get(key) is not None]
            if values:
                avg = sum(values) / len(values)
                print(f"Average {key}: {avg:.4f}")
        
        # Print metrics by fold
        if args.dataset_type == "cb":
            print("\nMetrics by fold:")
            for metrics in all_metrics:
                if 'fold' in metrics:
                    fold = metrics['fold']
                    print(f"Fold {fold}:")
                    for key in ['mse', 'rmse', 'nmse', 'r_rmse', 'r_pearson']:
                        if key in metrics:
                            print(f"  {key} = {metrics[key]:.4f}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()