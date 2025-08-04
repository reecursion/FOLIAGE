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
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# Global variables for model and tokenizer
model, tokenizer = None, None

def compute_metrics(pred):
    """Compute metrics for model evaluation."""
    global tokenizer, model
    references = pred.label_ids                  # shape: (batch_size, seq_len)
    generated_logits = pred.predictions          # shape: (batch_size, seq_len, vocab_size)

    print("References shape:", references.shape)
    print("Generated logits shape:", generated_logits.shape)

    # Get token IDs from logits by argmax over vocabulary dimension
    generated_ids = np.argmax(generated_logits, axis=-1)  # shape: (batch_size, seq_len)

    # For validation on the price prediction task
    nse = []

    for i in range(references.shape[0]):
        try:
            r_id = references[i]
            g_id = generated_ids[i]  # Fixed: use generated_ids instead of references

            # Remove padding from reference (-100)
            r_id = r_id[r_id != -100]
            g_id = g_id[g_id != -100]  # Fixed: remove padding from generated ids too

            # Decode
            reference = tokenizer.decode(r_id, skip_special_tokens=True)
            generated = tokenizer.decode(g_id, skip_special_tokens=True)  # Fixed: decode g_id
            
            # Extract prompt from the reference
            if "user" in reference and "assistant" in reference:
                # Split on user/assistant boundary for chatml-like templates
                parts = reference.split("assistant")
                prompt_part = parts[0]
                
                # Extract just the user content without template markers
                prompt = prompt_part.split("user")[-1].strip()
                
                # Format as messages for the chat template
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Get actual price from reference and predicted price from generated text
                actual_price = extract_final_price(reference)
                predicted_price = extract_final_price(generated)  # Fixed: use generated instead of generated_text
                
                # Calculate normalized metrics if both prices exist
                if predicted_price is not None and actual_price is not None:
                    err = (predicted_price - actual_price) ** 2 / actual_price
                    nse.append(err)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Return average error
    if len(nse) > 0:
        return {
            'nmse': np.mean(nse),
        }
    else:
        return {
            'nmse': float('nan'),
        }

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
    parser.add_argument("--eval_steps", type=int, default=1,
                        help="Steps between evaluations")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to save")
    
    return parser.parse_args()

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

def prepare_dataset(args):
    """Prepare dataset for finetuning with appropriate formatting."""
    global tokenizer
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
                # Make sure sale_price column exists
                if 'sale_price' not in group.columns:
                    print(f"Warning: sale_price column not found in dialogue {dialogue_id}, skipping")
                    continue
                    
                outcome = f"FINAL_PRICE: ${group['sale_price'].iloc[0]}"
                
                # Make sure target columns exist
                if 'buyer_target' not in group.columns or 'seller_target' not in group.columns:
                    print(f"Warning: target columns not found in dialogue {dialogue_id}, skipping")
                    continue
                    
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
                    "role": "user",  # Will map to "human" based on your mapping
                    "content": f"Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: [number]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}>"
                }]
                
                # Apply chat template - no tokenization, just formatting
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
                # Store dialogue info for evaluation
                dialogue_info[dialogue_id] = {
                    "buyer_target": buyer_target,
                    "seller_target": seller_target,
                    "sale_price": sale_price,
                    "prompt": input_text  # Store formatted prompt
                }
                
            elif args.dataset_type == "p4g":
                # Check for required columns
                if 'label' not in group.columns:
                    print(f"Warning: label column not found in dialogue {dialogue_id}, skipping")
                    continue
                
                # Get donation amount if available
                donation_amount = None
                if 'donation_amount' in group.columns:
                    donation_values = group['donation_amount'].dropna()
                    if len(donation_values) > 0:
                        donation_amount = donation_values.iloc[0]
                
                if donation_amount is not None and not pd.isna(donation_amount):
                    outcome = f"DONATION: YES, AMOUNT: ${donation_amount}"
                else:
                    outcome = "DONATION: YES" if group['label'].iloc[0].lower() == 'yes' else "DONATION: NO"
                
                # Create messages for chat template
                messages = [{
                    "role": "user",
                    "content": f"You are helping analyze a persuasion conversation.\n\nConversation:\n{formatted_conversation}{summary}\n\nPredict whether the persuadee will make a donation."
                }]
                
                # Apply chat template
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Create assistant message with outcome
            assistant_messages = [{
                "role": "assistant",
                "content": outcome
            }]
            
            # Format completion using chat template
            output_text = tokenizer.apply_chat_template(assistant_messages, tokenize=False, add_generation_prompt=False)
            
            # Complete text is the conversation with both user and assistant parts
            # For direct templating, combine the input and output
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

    else:
        # Dataset is already in the right format (one dialogue per row)
        print("Dataset is already in dialogue-level format")
        # Check for required columns
        if 'dialogue_id' not in df.columns or 'conversation' not in df.columns:
            raise ValueError("Dataset must contain 'dialogue_id' and 'conversation' columns")
            
        # Use chat template here too
        dataset_dict = {
            "dialogue_id": df["dialogue_id"].tolist(),
            "text": [tokenizer.apply_chat_template([{"role": "user", "content": row}], tokenize=False, add_generation_prompt=True) for row in df["conversation"].tolist()]
        }
        dialogue_info = {}
    
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
    global model, tokenizer
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize model and tokenizer first (only once) to ensure tokenizer is available for dataset preparation
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

    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )
    
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
    
    # Initialize empty DataFrame with required columns if file doesn't exist
    if not os.path.exists(predictions_file):
        pd.DataFrame(columns=[
            "dialogue_id", "fold", "prompt", "generated_text", "buyer_target", 
            "seller_target", "sale_price", "predicted_final_price", "success_sale", 
            "success_predicted", "normalized_squared_error"
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
        
        # Reset model for this fold - we keep the same tokenizer
        # Add LoRA adapters to fresh model for this fold
        model = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_rank,
            lora_dropout=0,
            bias="none",
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
            evaluation_strategy="steps",  # Fixed: eval_strategy to evaluation_strategy
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
            packing=True,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        print(f"Starting training for fold {fold+1}...")
        try:
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
                try:
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
                except Exception as e:
                    print(f"Error generating prediction for dialogue {dialogue_id}: {e}")
                    continue
            
            # Save predictions to CSV - append to existing file
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
                    try:
                        wandb.log({
                            f"fold_{fold+1}/price_nmse": price_nmse,
                            f"fold_{fold+1}/success_rmse": success_rmse
                        })
                        if "rpearson" in fold_result:
                            wandb.log({f"fold_{fold+1}/rpearson": fold_result["rpearson"]})
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
            if "price_nmse" in result:
                metrics_str += f", NMSE = {result['price_nmse']:.4f}, RMSE = {result['success_rmse']:.4f}"
                if "rpearson" in result:
                    metrics_str += f", Pearson = {result['rpearson']:.4f}"
            print(f"Fold {result['fold']}: {metrics_str}")
    else:
        print("No fold results to report.")
    
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
                try:
                    wandb.log({
                        "overall/price_nmse": overall_nmse,
                        "overall/success_rmse": overall_rmse,
                        "overall/rpearson": overall_pearson
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