import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                         BitsAndBytesConfig, TrainingArguments,
                         EarlyStoppingCallback)  # Added EarlyStoppingCallback
from trl import SFTTrainer

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess the logits before they're used in the metrics calculation.
    This prevents storing all logits in memory, which can cause OOM errors.
    """
    # Extract predicted IDs by taking the argmax along the vocabulary dimension
    pred_ids = torch.argmax(logits, dim=-1)
    # Return the predicted IDs and labels
    return pred_ids, labels

def compute_metrics(_, tokenizer, model, eval_dataset, dialogue_info, dataset_type):
    """
    Compute metrics for both CB and P4G datasets.
    
    Args:
        _: Unused eval_preds parameter (required by trainer interface)
        tokenizer: Tokenizer for encoding/decoding
        model: The current model being trained
        eval_dataset: The current fold's evaluation dataset
        dialogue_info: Dictionary with ground truth information
        dataset_type: Type of dataset (cb or p4g)
        
    Returns:
        Dictionary with metrics appropriate for the dataset type
    """
    # Initialize metric values
    total_metrics = {}
    valid_count = 0
    
    # Get a sample of dialogue IDs from the eval dataset
    # Limit to a small number for speed during training
    max_samples = max(10, len(eval_dataset))
    sample_indices = list(range(max_samples))
    
    # Set model to evaluation mode
    model.eval()
    
    # Process sampled examples
    for i in sample_indices:
        try:
            # Get dialogue ID
            dialogue_id = eval_dataset[i]["dialogue_id"]
            
            # Skip if dialogue_id not in dialogue_info
            if dialogue_id not in dialogue_info:
                continue
            
            # Get prompt from dialogue_info
            prompt = dialogue_info[dialogue_id]["prompt"]
            
            # Generate prediction
            with torch.no_grad():
                inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True)
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Process metrics based on dataset type
            if dataset_type == "cb":
                # Extract predicted price for Craigslist Bargain
                predicted_price = extract_final_price(pred_text)
                
                # Skip if no valid predicted price
                if predicted_price is None or predicted_price <= 0:
                    continue
                
                # Get ground truth
                sale_price = dialogue_info[dialogue_id]["sale_price"]
                
                # Calculate NMSE
                nmse = ((sale_price - predicted_price) ** 2) / sale_price
                
                # Update metrics
                if "nmse" not in total_metrics:
                    total_metrics["nmse"] = 0.0
                total_metrics["nmse"] += nmse
                
            elif dataset_type == "p4g":
                # Extract donation decision for Persuasion for Good
                pred_decision = extract_donation_decision(pred_text)
                
                # Skip if no valid prediction
                if pred_decision is None:
                    continue
                
                # Get ground truth
                true_decision = "YES" if dialogue_info[dialogue_id]["donation_made"] else "NO"
                
                # Calculate decision accuracy
                decision_correct = (pred_decision.lower() == true_decision.lower())
                
                # Update decision accuracy metric
                if "accuracy" not in total_metrics:
                    total_metrics["accuracy"] = 0.0
                total_metrics["accuracy"] += 1.0 if decision_correct else 0.0
            
            valid_count += 1
        
        except Exception as e:
            # Skip problematic examples
            continue
    
    model.train()
    
    # Calculate final metrics
    result = {}
    if valid_count > 0:
        if dataset_type == "cb" and "nmse" in total_metrics:
            result["nmse"] = total_metrics["nmse"] / valid_count
        elif dataset_type == "p4g" and "accuracy" in total_metrics:
            result["accuracy"] = total_metrics["accuracy"] / valid_count
    
    return result

def get_compute_metrics_fn(tokenizer, model, eval_dataset, dialogue_info, dataset_type):
    """Creates compute_metrics function with necessary context"""
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer, model, eval_dataset, dialogue_info, dataset_type)
    return compute_metrics_wrapper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune Llama 3.1 8B on negotiation datasets with k-fold cross validation")
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["cb", "p4g"],
                        help="Type of dataset (cb for Craigslist Bargain, p4g for Persuasion for Good)")
    # Intentions arguments
    parser.add_argument("--scaffolding_type", type=str, required=True, choices=["local", "global", "both", "none"],
                        help="Type of intentions to use: local (only intentions), global (only summaries), both, or none")
    parser.add_argument("--summary_type", type=str, default="none",
                        choices=["none", "traditional", "scd", "relational", "scm", "appraisal_theory", "politeness_theory_stage2"],
                        help="Type of summary to use for global intentions")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/data/user_data/gganeshl/output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    # K-fold arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval_steps", type=int, default=1, help="Steps between evaluations")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to save")
    # Early stopping arguments (Added)
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Threshold for early stopping")
    
    return parser.parse_args()


def extract_donation_decision(text):
    """Extract donation decision from model output for P4G dataset."""
    try:
        # First check for DONATION: YES/NO format
        donation_match = re.search(r'DONATION:\s*(YES|NO)', text, re.IGNORECASE)
        if donation_match:
            donation_decision = donation_match.group(1).upper()
            return donation_decision
        return None
    except:
        return None

def extract_final_price(text):
    """Extract the final price from model output."""
    try:
        # First try the FINAL_PRICE format
        match = re.search(r'FINAL_PRICE:\s*\$?(\d+\.?\d*)', text)
        if match:
            return float(match.group(1))
        
        # If no match, try looking for any number with dollar sign
        match = re.search(r'\$(\d+\.?\d*)', text)
        if match:
            return float(match.group(1))
        
        # If still no match, try to find any number
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            return float(match.group(1))
            
        return None
    except:
        return None


def prepare_dataset(args, tokenizer):
    """Prepare dataset for finetuning with appropriate formatting."""
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    print("Processing utterance-level dataset...")
    # Group by dialogue_id
    grouped_data = []
    dialogue_info = {}  # Store dialogue info for evaluation
        
    for dialogue_id, group in df.groupby('dialogue_id'):
        # Sort by utterance index
        group = group.sort_values('utterance_idx')
        
        conversation = []
        for _, row in group.iterrows():
            # Format with or without intentions based on scaffolding type
            utterance = row['utterance']

            # Include speaker role context for P4G dataset
            speaker = row['speaker']
            if args.dataset_type == "p4g":
                if speaker == "EE":
                    speaker = "Persuadee"
                elif speaker == "ER":
                    speaker = "Persuader"

            if args.scaffolding_type in ["local", "both"] and 'intention' in row and pd.notna(row['intention']):
                # Include intention for local scaffolding
                conversation.append(f"{speaker}: {utterance} [{row['intention']}]")
            else:
                # No intentions
                conversation.append(f"{speaker}: {utterance}")
        
        formatted_conversation = "\n".join(conversation)
        
        # Add summary if using global scaffolding
        summary = ""
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            summary_column = f"{args.summary_type}_summary"
            if summary_column in group.columns and pd.notna(group[summary_column].iloc[0]):
                summary = f", [Summary: {group[summary_column].iloc[0]}]"
        
        if args.dataset_type == "cb":
            # Make sure required columns exist
            required_columns = ['sale_price', 'buyer_target', 'seller_target']
            if not all(col in group.columns for col in required_columns):
                print(f"Warning: Required columns missing in dialogue {dialogue_id}, skipping")
                continue
                
            outcome = f"FINAL_PRICE: ${group['sale_price'].iloc[0]}"
            buyer_target = group['buyer_target'].iloc[0]
            seller_target = group['seller_target'].iloc[0]
            sale_price = group['sale_price'].iloc[0]
            
            # Always use "with intentions" if they're included
            intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
            
            # Format summary part based on global scaffolding
            summary_part = ""
            if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
                summary_part = f", [summary]"
            
            # Create messages for chat template
            messages = [{
                "role": "user",  
                "content": f"Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: [number]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}>"
            }]
            
            # Apply chat template - no tokenization, just formatting
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Create assistant message with outcome
            assistant_messages = [{
                "role": "assistant",
                "content": outcome
            }]
            
            # Store dialogue info for evaluation
            dialogue_info[dialogue_id] = {
                "buyer_target": buyer_target,
                "seller_target": seller_target,
                "sale_price": sale_price,
                "prompt": input_text
            }
            
        elif args.dataset_type == "p4g":
            # Check for required columns
            if 'donation_made' not in group.columns:
                print(f"Warning: donation_made column not found in dialogue {dialogue_id}, skipping")
                continue
            
            # Get donation information
            donation_made = bool(group['donation_made'].iloc[0])
            
            if donation_made:
                outcome = "DONATION: YES"
            else:
                outcome = "DONATION: NO"
            
            # Always use "with intentions" if they're included
            intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
            
            # Format summary part based on global scaffolding
            summary_part = ""
            if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
                summary_part = f" {summary}"
            
            # Create messages for chat template
            messages = [{
                "role": "user",
                "content": f"You are helping analyze a persuasion conversation{intentions_note}. Predict whether the persuadee will make a donation on the spot at the end of this conversation. Provide your answer in the format 'DONATION: YES/NO'\n\nConversation:\n{formatted_conversation}{summary_part}"
            }]
            
            # Apply chat template
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Create assistant message with outcome
            assistant_messages = [{
                "role": "assistant",
                "content": outcome
            }]
            
            # Store dialogue info for evaluation
            dialogue_info[dialogue_id] = {
                "donation_made": donation_made,
                "prompt": input_text
            }
        # Format completion using chat template
        output_text = tokenizer.apply_chat_template(assistant_messages, tokenize=False, add_generation_prompt=False)
        
        # Complete text is the conversation with both user and assistant parts
        full_text = input_text + output_text
        
        # Append to grouped data
        grouped_data.append({
            "dialogue_id": dialogue_id,
            "input": input_text,
            "output": output_text,
            "text": full_text
        })
    
    dataset_dict = {
        "dialogue_id": [item["dialogue_id"] for item in grouped_data],
        "input": [item["input"] for item in grouped_data],
        "output": [item["output"] for item in grouped_data],
        "text": [item["text"] for item in grouped_data]
    }
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Show a sample
    if len(dataset) > 0:
        print("\nSample input-output pair:")
        sample_idx = 0
        if 'input' in dataset[sample_idx] and 'output' in dataset[sample_idx]:
            print(f"INPUT:\n{dataset[sample_idx]['input']}")
            print(f"OUTPUT:\n{dataset[sample_idx]['output']}")
    
    return dataset, dialogue_info


def perform_kfold_cross_validation(args):
    """Perform k-fold cross validation."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    full_dataset, dialogue_info = prepare_dataset(args, tokenizer)
    
    # Initialize KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Track metrics across folds
    fold_results = []
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset_type == "cb":
        results_dir = "/home/gganeshl/FOLIAGE/src/sft/results/craigslistbargain"
    elif args.dataset_type == "p4g":
        results_dir = "/home/gganeshl/FOLIAGE/src/sft/results/persuasionforgood"
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract ratio from filename
    ratio_match = re.search(r'ratio_(\d+\.\d+)', args.dataset_path)
    ratio = ratio_match.group(1) if ratio_match else "unknown"
    
    # Create experiment name for wandb
    experiment_name = f"{args.dataset_type}_ratio_{ratio}_{args.scaffolding_type}"
    if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
        experiment_name += f"_{args.summary_type}"
    
    # Initialize predictions CSV
    predictions_file = os.path.join(results_dir, f"{experiment_name}_predictions.csv")
    
    # Initialize empty DataFrame if file doesn't exist
    if not os.path.exists(predictions_file):
        if args.dataset_type == "cb":
            pd.DataFrame(columns=[
                "dialogue_id", "fold", "prompt", "generated_text", "buyer_target", 
                "seller_target", "sale_price", "predicted_final_price", "success_sale", 
                "success_predicted", "normalized_squared_error"
            ]).to_csv(predictions_file, index=False)
        elif args.dataset_type == "p4g":
            pd.DataFrame(columns=[
                "dialogue_id", "fold", "prompt", "generated_text", 
                "donation_made", "predicted_decision", "decision_correct"
            ]).to_csv(predictions_file, index=False)
    
    # Initialize wandb
    try:
        wandb.init(project=experiment_name, name=experiment_name)
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
    
    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold+1}/{args.n_folds}")
        print(f"{'='*50}")
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Convert numpy indices to python ints
        train_idx = [int(i) for i in train_idx]
        val_idx = [int(i) for i in val_idx]
        
        # Split dataset into train and validation
        train_dataset = full_dataset.select(train_idx)
        val_dataset = full_dataset.select(val_idx)
        
        print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")
        
        # Initialize model with LoRA for this fold
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        
        # Set up LoRA configuration
        rank = args.rank
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=rank*2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        )
        
        # Set up training arguments
        model_checkpoint_path = os.path.join(fold_output_dir, "checkpoints")
        
        # Set metric for best model based on dataset type
        metric_for_best_model = "eval_nmse" if args.dataset_type == "cb" else "eval_accuracy"
        greater_is_better = False if args.dataset_type == "cb" else True
        
        training_arguments = TrainingArguments(
            output_dir=model_checkpoint_path,
            optim='paged_adamw_32bit',
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            log_level='debug',
            eval_strategy="steps",
            save_strategy='steps',
            logging_steps=8,
            eval_steps=0.1,
            save_steps=0.1,
            learning_rate=1e-4,
            fp16=True,
            num_train_epochs=args.epochs,
            # max_steps=120,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            lr_scheduler_type='linear',
            save_total_limit=2,  
            metric_for_best_model=metric_for_best_model,  
            greater_is_better=greater_is_better,
            eval_accumulation_steps=1,
        )
        
        # Initialize early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,  # Use patience of 3
            early_stopping_threshold=args.early_stopping_threshold  # Use threshold of 0.001
        )

        class CustomSFTTrainer(SFTTrainer):
            def __init__(self, *args, preprocess_logits_for_metrics=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        
        # Initialize compute_metrics function with model, eval_dataset and dialogue_info
        compute_metrics_fn = get_compute_metrics_fn(tokenizer, model, val_dataset, dialogue_info, args.dataset_type)

        # Initialize SFTTrainer with the early stopping callback
        trainer = CustomSFTTrainer(
            model=model,
            dataset_text_field="text",
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            max_seq_length=args.max_length,
            tokenizer=tokenizer,
            args=training_arguments,
            compute_metrics=compute_metrics_fn,
            callbacks=[early_stopping_callback],  # Add early stopping callback
        )
        
        trainer.model.print_trainable_parameters()
        
        # Train and evaluate
        try:
            print(f"Starting training for fold {fold+1}...")
            train_result = trainer.train()
            
            print(f"Evaluating fold {fold+1}...")
            eval_result = trainer.evaluate()
            
            # Save results for this fold
            fold_result = {
                "fold": fold + 1,
                "train_loss": train_result.training_loss,
                "eval_loss": eval_result["eval_loss"],
            }
            
            # Save the model
            model_path = os.path.join(fold_output_dir, "final_model")
            trainer.save_model(model_path)
            
            # Evaluate on validation set
            print(f"Running detailed evaluation on fold {fold+1}...")
            
            # Get validation dialogue IDs
            val_dialogue_ids = [full_dataset[i]["dialogue_id"] for i in val_idx]
            
            # Run predictions
            fold_predictions = []
            for dialogue_id in val_dialogue_ids:
                if dialogue_id not in dialogue_info:
                    continue
                    
                # Get prompt 
                prompt = dialogue_info[dialogue_id]["prompt"]
                
                # Generate prediction
                try:
                    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
                    generated_text = tokenizer.decode(outputs[0])
                    
                    # Process based on dataset type
                    if args.dataset_type == "cb":
                        # Extract predicted price
                        predicted_price = extract_final_price(generated_text)
                        
                        # Add to predictions
                        prediction = {
                            "dialogue_id": dialogue_id,
                            "fold": fold + 1,
                            "prompt": prompt,
                            "generated_text": generated_text,
                            "buyer_target": dialogue_info[dialogue_id]["buyer_target"],
                            "seller_target": dialogue_info[dialogue_id]["seller_target"],
                            "sale_price": dialogue_info[dialogue_id]["sale_price"],
                            "predicted_final_price": predicted_price
                        }
                        
                        # Calculate metrics if predicted price exists
                        if predicted_price is not None:
                            buyer_target = dialogue_info[dialogue_id]["buyer_target"]
                            seller_target = dialogue_info[dialogue_id]["seller_target"]
                            sale_price = dialogue_info[dialogue_id]["sale_price"]
                            
                            # Calculate success metrics
                            prediction["success_sale"] = (sale_price - buyer_target) / (seller_target - buyer_target)
                            prediction["success_predicted"] = (predicted_price - buyer_target) / (seller_target - buyer_target)
                            
                            # Calculate NMSE
                            prediction["normalized_squared_error"] = ((sale_price - predicted_price) ** 2) / sale_price
                    
                    elif args.dataset_type == "p4g":
                        # Extract donation decision
                        pred_decision = extract_donation_decision(generated_text)
                        
                        # Add to predictions
                        prediction = {
                            "dialogue_id": dialogue_id,
                            "fold": fold + 1,
                            "prompt": prompt,
                            "generated_text": generated_text,
                            "donation_made": dialogue_info[dialogue_id]["donation_made"],
                            "predicted_decision": pred_decision
                        }
                        
                        # Calculate metrics if predicted decision exists
                        if pred_decision is not None:
                            true_decision = "YES" if dialogue_info[dialogue_id]["donation_made"] else "NO"
                            prediction["decision_correct"] = (pred_decision.lower() == true_decision.lower())
                    
                    fold_predictions.append(prediction)
                except Exception as e:
                    print(f"Error generating prediction for dialogue {dialogue_id}: {e}")
                    continue
            
            # Save predictions to CSV
            if fold_predictions:
                try:
                    # Read existing predictions if any
                    try:
                        existing_df = pd.read_csv(predictions_file)
                        # Filter out predictions from this fold if they already exist
                        existing_df = existing_df[existing_df["fold"] != fold + 1]
                        # Append new predictions
                        new_df = pd.DataFrame(fold_predictions)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df.to_csv(predictions_file, index=False)
                    except:
                        # If file doesn't exist or is empty, create new
                        pd.DataFrame(fold_predictions).to_csv(predictions_file, index=False)
                except Exception as e:
                    print(f"Error saving predictions: {e}")
            
            # Calculate metrics for this fold based on dataset type
            if fold_predictions:
                df = pd.DataFrame(fold_predictions)
                
                if args.dataset_type == "cb":
                    df = df.dropna(subset=['predicted_final_price', 'success_sale', 'success_predicted'])
                    
                    if len(df) > 0:
                        # Calculate metrics
                        price_nmse = df["normalized_squared_error"].mean()
                        success_rmse = np.sqrt(mean_squared_error(df['success_sale'], df['success_predicted']))
                        
                        # Calculate Pearson correlation
                        if len(df) > 1:
                            rpearson, _ = pearsonr(df['success_sale'], df['success_predicted'])
                            fold_result["rpearson"] = rpearson
                        
                        # Add metrics to fold result
                        fold_result["price_nmse"] = price_nmse
                        fold_result["success_rmse"] = success_rmse
                        
                        # Log to wandb
                        try:
                            wandb.log({
                                f"fold_{fold+1}/price_nmse": price_nmse,
                                f"fold_{fold+1}/success_rmse": success_rmse
                            })
                            if "rpearson" in fold_result:
                                wandb.log({f"fold_{fold+1}/rpearson": fold_result["rpearson"]})
                        except:
                            print("Warning: Could not log to wandb")
                
                elif args.dataset_type == "p4g":
                    df = df.dropna(subset=['predicted_decision', 'donation_made'])
                    
                    if len(df) > 0:
                        accuracy = df['decision_correct'].mean()
                        
                        # Add metrics to fold result
                        fold_result["accuracy"] = accuracy
                        
                        # Log to wandb
                        try:
                            wandb.log({
                                f"fold_{fold+1}/accuracy": accuracy
                            })
                        except:
                            print("Warning: Could not log to wandb")
            
            fold_results.append(fold_result)
            
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            continue
        finally:
            # Clean up GPU memory
            del model
            del trainer
            torch.cuda.empty_cache()
    
    # Print cross-validation summary
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    
    if fold_results:
        # Calculate aggregate metrics
        mean_train_loss = np.mean([result["train_loss"] for result in fold_results])
        mean_eval_loss = np.mean([result["eval_loss"] for result in fold_results])
        
        print(f"Mean Training Loss: {mean_train_loss:.4f}")
        print(f"Mean Evaluation Loss: {mean_eval_loss:.4f}")
        
        # Print individual fold results
        for result in fold_results:
            metrics_str = f"Train Loss = {result['train_loss']:.4f}, Eval Loss = {result['eval_loss']:.4f}"
            
            if args.dataset_type == "cb":
                if "price_nmse" in result:
                    metrics_str += f", NMSE = {result['price_nmse']:.4f}, RMSE = {result['success_rmse']:.4f}"
                    if "rpearson" in result:
                        metrics_str += f", Pearson = {result['rpearson']:.4f}"
            elif args.dataset_type == "p4g":
                if "accuracy" in result:
                    metrics_str += f", Accuracy = {result['accuracy']:.4f}"
            
            print(f"Fold {result['fold']}: {metrics_str}")
    else:
        print("No fold results to report.")
    
    # Calculate overall metrics from saved predictions
    try:
        all_predictions = pd.read_csv(predictions_file)
        
        if args.dataset_type == "cb":
            all_predictions = all_predictions.dropna(subset=['predicted_final_price', 'success_sale', 'success_predicted'])
            
            if len(all_predictions) > 0:
                overall_nmse = all_predictions["normalized_squared_error"].mean()
                overall_rmse = np.sqrt(mean_squared_error(
                    all_predictions['success_sale'], all_predictions['success_predicted']))
                
                print(f"\nOverall Metrics:")
                print(f"NMSE = {overall_nmse:.4f}, RMSE = {overall_rmse:.4f}")
                
                # Calculate overall Pearson correlation
                if len(all_predictions) > 1:
                    overall_pearson, _ = pearsonr(
                        all_predictions['success_sale'], all_predictions['success_predicted'])
                    print(f"Pearson = {overall_pearson:.4f}")
                    
                    # Log overall metrics to wandb
                    try:
                        wandb.log({
                            "overall/price_nmse": overall_nmse,
                            "overall/success_rmse": overall_rmse,
                            "overall/rpearson": overall_pearson
                        })
                    except:
                        print("Warning: Could not log to wandb")
        
        elif args.dataset_type == "p4g":
            all_predictions = all_predictions.dropna(subset=['predicted_decision', 'donation_made'])
            
            if len(all_predictions) > 0:
                # Ensure decision_correct is properly calculated
                all_predictions['true_decision'] = all_predictions['donation_made'].apply(
                    lambda x: "YES" if x else "NO")
                
                # Calculate overall accuracy
                if 'decision_correct' not in all_predictions.columns:
                    all_predictions['decision_correct'] = all_predictions.apply(
                        lambda row: row['predicted_decision'].lower() == row['true_decision'].lower(), axis=1)
                
                overall_accuracy = all_predictions['decision_correct'].mean()
                
                print(f"\nOverall Metrics:")
                print(f"Accuracy = {overall_accuracy:.4f}")
                
                # Log overall metrics to wandb
                try:
                    wandb.log({
                        "overall/accuracy": overall_accuracy
                    })
                except:
                    print("Warning: Could not log to wandb")
    except Exception as e:
        print(f"Error calculating overall metrics: {e}")
    
    try:
        wandb.finish()
    except:
        pass
    
    return fold_results

def main():
    args = parse_arguments()
    perform_kfold_cross_validation(args)


if __name__ == "__main__":
    main()