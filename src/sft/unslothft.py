import argparse
import pandas as pd
import os
import torch
import re
import numpy as np
import wandb
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, TextStreamer
from unsloth import is_bfloat16_supported

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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank parameter")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    # K-fold arguments
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for cross validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Steps between evaluations")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to save")
    
    return parser.parse_args()

def extract_final_price(text):
    """Extract the final price from model output."""
    match = re.search(r'FINAL_PRICE:\s*\$?(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None

def prepare_dataset(args):
    """Prepare dataset for finetuning with appropriate formatting."""
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    # Check if dataset is in utterance level format (needs grouping)
    is_utterance_level = all(col in df.columns for col in ['dialogue_id', 'utterance_idx', 'speaker', 'utterance'])
    
    if is_utterance_level:
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
                if args.scaffolding_type in ["local", "both"] and 'intention' in row and pd.notna(row['intention']):
                    # Include intention for local scaffolding
                    conversation.append(f"{row['speaker']}: {utterance} [{row['intention']}]")
                else:
                    # No intentions
                    conversation.append(f"{row['speaker']}: {utterance}")
            
            formatted_conversation = "\n".join(conversation)
            
            # Add summary if using global scaffolding
            summary = ""
            if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
                summary_column = f"{args.summary_type}_summary"
                if summary_column in group.columns and pd.notna(group[summary_column].iloc[0]):
                    summary = f", [Summary: {group[summary_column].iloc[0]}]"
            
            # Get outcome/label
            if args.dataset_type == "cb":
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
                
                # Use the exact prompt template
                input_text = f"<s>[INST] Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: [number]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}> [/INST]"
                
                # Store dialogue info for evaluation
                dialogue_info[dialogue_id] = {
                    "buyer_target": buyer_target,
                    "seller_target": seller_target,
                    "sale_price": sale_price,
                    "prompt": input_text
                }
                
            elif args.dataset_type == "p4g":
                # Get donation amount if available
                donation_amount = group.get('donation_amount', None)
                if donation_amount is not None and not pd.isna(donation_amount.iloc[0]):
                    donation_value = donation_amount.iloc[0]
                    outcome = f"DONATION: YES, AMOUNT: ${donation_value}"
                else:
                    outcome = "DONATION: YES" if group['label'].iloc[0].lower() == 'yes' else "DONATION: NO"
                
                # Keep original p4g prompt format
                input_text = f"<s>[INST] You are helping analyze a persuasion conversation.\n\nConversation:\n{formatted_conversation}{summary}\n\nPredict whether the persuadee will make a donation. [/INST]"
            
            # Format completion (target)
            output_text = f" {outcome} </s>"
            
            # Combined for training
            grouped_data.append({
                "dialogue_id": dialogue_id,
                "input": input_text,
                "output": output_text,
                "text": input_text + output_text  
            })
        
        dataset_dict = {
            "dialogue_id": [item["dialogue_id"] for item in grouped_data],
            "input": [item["input"] for item in grouped_data],
            "output": [item["output"] for item in grouped_data],
            "text": [item["text"] for item in grouped_data]
        }
        
    else:
        # Dataset is already in the right format (one dialogue per row)
        print("Dataset is already in dialogue-level format")
        dataset_dict = {
            "dialogue_id": df["dialogue_id"].tolist(),
            "text": ["<s>[INST] " + row + " [/INST] Output </s>" for row in df["conversation"].tolist()]
        }
        dialogue_info = {}
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Show a sample
    print("\nSample input-output pair:")
    sample_idx = 0
    print(f"INPUT:\n{dataset[sample_idx]['input']}")
    print(f"OUTPUT:\n{dataset[sample_idx]['output']}")
    
    return dataset, dialogue_info

def perform_kfold_cross_validation(args):
    """Perform k-fold cross validation."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare the dataset
    full_dataset, dialogue_info = prepare_dataset(args)
    
    # Initialize KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Track metrics across folds
    fold_results = []
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = "/home/gganeshl/FOLIAGE/src/sft/results/craigslistbargain"
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
    if not os.path.exists(predictions_file):
        # Create empty DataFrame with required columns
        pd.DataFrame(columns=[
            "dialogue_id", "fold", "prompt", "generated_text", "buyer_target", 
            "seller_target", "sale_price", "predicted_final_price", "success_sale", 
            "success_predicted", "normalized_squared_error"
        ]).to_csv(predictions_file, index=False)
    
    # Initialize wandb
    wandb.init(project=experiment_name, name=experiment_name)
    
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
        
        # Initialize model and tokenizer for this fold
        print("Initializing model and tokenizer...")
        max_seq_length = args.max_length
        dtype = None  # Auto detect
        load_in_4bit = True
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Setup training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=fold_output_dir,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
        )
        
        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            data_collator=data_collator,
        )
        
        # Train the model
        print(f"Starting training for fold {fold+1}...")
        train_result = trainer.train()
        
        # Evaluate
        print(f"Evaluating fold {fold+1}...")
        eval_result = trainer.evaluate()
        
        # Save results
        fold_result = {
            "fold": fold + 1,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
        }
        
        # Save the model for this fold
        model_path = os.path.join(fold_output_dir, "final_model")
        trainer.save_model(model_path)
        
        # Evaluation on validation set for metrics
        print(f"Running detailed evaluation on fold {fold+1}...")
        FastLanguageModel.for_inference(model)
        
        # Get validation dialogue IDs
        val_dialogue_ids = [full_dataset[i]["dialogue_id"] for i in val_idx]
        
        # Evaluate on validation set
        fold_predictions = []
        for dialogue_id in val_dialogue_ids:
            if dialogue_id not in dialogue_info:
                continue
                
            # Get prompt 
            prompt = dialogue_info[dialogue_id]["prompt"]
            
            # Generate prediction
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
            generated_text = tokenizer.decode(outputs[0])
            
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
            
            # Calculate normalized metrics if predicted price exists
            if predicted_price is not None:
                buyer_target = dialogue_info[dialogue_id]["buyer_target"]
                seller_target = dialogue_info[dialogue_id]["seller_target"]
                sale_price = dialogue_info[dialogue_id]["sale_price"]
                
                # Calculate success metrics
                prediction["success_sale"] = (sale_price - buyer_target) / (seller_target - buyer_target)
                prediction["success_predicted"] = (predicted_price - buyer_target) / (seller_target - buyer_target)
                
                # Calculate NMSE
                prediction["normalized_squared_error"] = ((sale_price - predicted_price) ** 2) / sale_price
            
            fold_predictions.append(prediction)
        
        # Save predictions to CSV - append to existing file
        if fold_predictions:
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
        
        # Calculate metrics for this fold's predictions
        if fold_predictions:
            df = pd.DataFrame(fold_predictions)
            df = df.dropna(subset=['predicted_final_price', 'success_sale', 'success_predicted'])
            
            if len(df) > 0:
                # Calculate metrics
                price_nmse = df["normalized_squared_error"].mean()
                success_rmse = np.sqrt(mean_squared_error(df['success_sale'], df['success_predicted']))
                
                # Calculate Pearson correlation
                if len(df) > 1:  # Need at least 2 points for correlation
                    rpearson, _ = pearsonr(df['success_sale'], df['success_predicted'])
                    fold_result["rpearson"] = rpearson
                
                # Add metrics to fold result
                fold_result["price_nmse"] = price_nmse
                fold_result["success_rmse"] = success_rmse
                
                # Log to wandb
                wandb.log({
                    f"fold_{fold+1}/price_nmse": price_nmse,
                    f"fold_{fold+1}/success_rmse": success_rmse
                })
                if "rpearson" in fold_result:
                    wandb.log({f"fold_{fold+1}/rpearson": fold_result["rpearson"]})
        
        fold_results.append(fold_result)
        
        # Clean up GPU memory
        del model
        del trainer
        torch.cuda.empty_cache()
    
    # Print cross-validation summary
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    
    # Calculate aggregate metrics
    mean_train_loss = np.mean([result["train_loss"] for result in fold_results])
    mean_eval_loss = np.mean([result["eval_loss"] for result in fold_results])
    
    print(f"Mean Training Loss: {mean_train_loss:.4f}")
    print(f"Mean Evaluation Loss: {mean_eval_loss:.4f}")
    
    # Print individual fold results
    for result in fold_results:
        metrics_str = f"Train Loss = {result['train_loss']:.4f}, Eval Loss = {result['eval_loss']:.4f}"
        if "price_nmse" in result:
            metrics_str += f", NMSE = {result['price_nmse']:.4f}, RMSE = {result['success_rmse']:.4f}"
            if "rpearson" in result:
                metrics_str += f", Pearson = {result['rpearson']:.4f}"
        print(f"Fold {result['fold']}: {metrics_str}")
    
    # Calculate overall metrics from saved predictions
    try:
        all_predictions = pd.read_csv(predictions_file)
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
                wandb.log({
                    "overall/price_nmse": overall_nmse,
                    "overall/success_rmse": overall_rmse,
                    "overall/rpearson": overall_pearson
                })
    except Exception as e:
        print(f"Error calculating overall metrics: {e}")
    
    wandb.finish()
    
    return fold_results

def main():
    args = parse_arguments()
    perform_kfold_cross_validation(args)

if __name__ == "__main__":
    main()