import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

def compute_metrics(eval_preds, val_idx=None, full_dataset=None):
    """
    Compute classification metrics for evaluation.
    
    Args:
        eval_preds: Tuple containing predictions and labels
        val_idx: Indices of validation examples
        full_dataset: The full dataset
        
    Returns:
        Dictionary with metrics
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # # Print first 3 examples if validation indices and dataset are provided
    # if val_idx is not None and full_dataset is not None:
    #     print("\n----- EVALUATION PREDICTIONS FOR FIRST 3 EXAMPLES -----")
    #     for i in range(min(3, len(predictions))):
    #         example_idx = val_idx[i] if i < len(val_idx) else i
    #         original_text = full_dataset[example_idx]["original_text"]
    #         # Truncate text for readability
    #         truncated_text = original_text[:200] + "..." if len(original_text) > 200 else original_text
            
    #         print(f"Example {i+1}:")
    #         print(f"Text: {truncated_text}")
    #         print(f"True label: {labels[i]} ")
    #         print(f"Predicted label: {predictions[i]}")
    #         print(f"Correct: {labels[i] == predictions[i]}")
    #         print("-" * 50)
    
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))
    unique, counts = np.unique(predictions, return_counts=True)
    print("Prediction distribution:", dict(zip(unique, counts)))
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune Llama 3.1 8B on negotiation datasets with k-fold cross validation")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["p4g"],
                        help="Type of dataset (p4g for Persuasion for Good)")
    
    # Intentions arguments
    parser.add_argument("--scaffolding_type", type=str, required=True, choices=["local", "global", "both", "none"],
                        help="Type of intentions to use: local (only intentions), global (only summaries), both, or none")
    parser.add_argument("--summary_type", type=str, default="none",
                        choices=["none", "traditional", "scd", "relational", "scm", "appraisal_theory", "politeness_theory_stage2"],
                        help="Type of summary to use for global intentions")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    
    # K-fold arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval_steps", type=float, default=0.1, help="Steps between evaluations")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to save")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Threshold for early stopping")
    
    return parser.parse_args()

def prepare_dataset(args, tokenizer):
    """Prepare dataset for classification with appropriate formatting."""
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    if args.dataset_type != "p4g":
        print(f"This script currently only supports the p4g dataset type. You specified: {args.dataset_type}")
        return None
    
    print("Processing dialogue-level dataset...")
    # Group by dialogue_id
    grouped_data = []
    
    for dialogue_id, group in df.groupby('dialogue_id'):
        # Sort by utterance index
        group = group.sort_values('utterance_idx')
        
        # Check if donation_made column exists
        if 'donation_made' not in group.columns:
            print(f"Warning: donation_made column not found in dialogue {dialogue_id}, skipping")
            continue
        
        # Get donation information (label)
        donation_made = bool(group['donation_made'].iloc[0])
        label = 1 if donation_made else 0  # Convert to integer labels
        
        # Format conversation
        conversation = []
        for _, row in group.iterrows():
            # Format speaker role
            speaker = row['speaker']
            if speaker == "EE":
                speaker = "Persuadee"
            elif speaker == "ER":
                speaker = "Persuader"
            
            # Format with or without intentions based on scaffolding type
            utterance = row['utterance']
            if args.scaffolding_type in ["local", "both"] and 'intention' in row and pd.notna(row['intention']):
                # Include intention for local scaffolding
                conversation.append(f"{speaker}: {utterance} [{row['intention']}]")
            else:
                # No intentions
                conversation.append(f"{speaker}: {utterance}")
        
        formatted_conversation = "\n".join(conversation)
        
        # Add summary if using global scaffolding
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            summary_column = f"{args.summary_type}_summary"
            if summary_column in group.columns and pd.notna(group[summary_column].iloc[0]):
                summary = f"Summary: {group[summary_column].iloc[0]}"
                formatted_conversation = f"{formatted_conversation}\n{summary}"  # Add newline before summary
        
        # Create a prompt for classification
        intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
        prompt = f"You are helping analyze a persuasion conversation{intentions_note}. Predict whether the persuadee will make a donation on the spot at the end of this conversation. Provide your answer in the format 'DONATION: YES/NO'\n\nConversation:\n{formatted_conversation}"
        
        # Add to dataset
        grouped_data.append({
            "dialogue_id": dialogue_id,
            "text": prompt,
            "label": label,  # Use integer labels: 0 for NO, 1 for YES
            "original_text": formatted_conversation
        })
    
    dataset_dict = {
        "dialogue_id": [item["dialogue_id"] for item in grouped_data],
        "text": [item["text"] for item in grouped_data],
        "label": [item["label"] for item in grouped_data],
        "original_text": [item["original_text"] for item in grouped_data]
    }
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Show a sample
    if len(tokenized_dataset) > 0:
        print("\nSample input-label pair:")
        sample_idx = 0
        print(f"TEXT:\n{tokenized_dataset[sample_idx]['text']}")
        print(f"DONATED: {tokenized_dataset[sample_idx]['label']} ({'YES' if tokenized_dataset[sample_idx]['label'] == 1 else 'NO'})")
    
    return tokenized_dataset


def perform_kfold_cross_validation(args):
    """Perform k-fold cross validation for classification."""
    # Check dataset type
    if args.dataset_type != "p4g":
        print(f"This script currently only supports the p4g dataset type. You specified: {args.dataset_type}")
        return []
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    
    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare the dataset
    full_dataset = prepare_dataset(args, tokenizer)
    if full_dataset is None:
        return []
    
    # Initialize KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Track metrics across folds
    fold_results = []
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, "results", args.dataset_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract ratio from filename
    ratio_match = re.search(r'ratio_(\d+\.\d+)', args.dataset_path)
    ratio = ratio_match.group(1) if ratio_match else "unknown"
    
    # Create experiment name for wandb
    experiment_name = f"{args.dataset_type}_classification_ratio_{ratio}_{args.scaffolding_type}"
    if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
        experiment_name += f"_{args.summary_type}"
    
    # Initialize predictions CSV
    predictions_file = os.path.join(f"/home/gganeshl/FOLIAGE/src/sft/results/{args.dataset_type}", f"{experiment_name}_predictions.csv")
    
    # Initialize empty DataFrame if file doesn't exist
    if not os.path.exists(predictions_file):
        pd.DataFrame(columns=[
            "dialogue_id", "fold", "text", "label", "predicted_label", 
            "correct", "confidence_score"
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
        
        # Calculate class weights for balanced training
        train_labels = [train_dataset[i]["label"] for i in range(len(train_dataset))]
        class_counts = np.bincount(train_labels)
        
        # Ensure we have counts for all classes (even if some are zero)
        if len(class_counts) < 2:
            class_counts = np.pad(class_counts, (0, 2 - len(class_counts)), 'constant')
        
        print(f"Raw class counts: {class_counts}")
        
        # Calculate weights and normalize
        class_weights = 1.0 / np.maximum(class_counts, 1)  # Avoid division by zero
        class_weights = class_weights / np.sum(class_weights)
        
        # Ensure class weights are float32 to match model dtype
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        print(f"Class weights tensor type: {class_weights_tensor.dtype}")
        
        print(f"Class distribution in training set: {class_counts}")
        print(f"Class weights: {class_weights}")
        
        # Initialize model with QLoRA for this fold
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float32,  # Changed from float16 to float32
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            num_labels=2,  # Binary classification
            device_map="auto"
        )
        
        # Set pad_token_id in the model config
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Prepare model for training with LoRA
        # Instead of using prepare_model_for_kbit_training, directly use get_peft_model
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank*2,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        )
        
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        
        # Set up training arguments
        model_checkpoint_path = os.path.join(fold_output_dir, "checkpoints")
        
        # Convert eval_steps from float to int or use fraction
        if isinstance(args.eval_steps, float) and args.eval_steps < 1.0:
            # Keep as float for automatic calculation based on dataset size
            eval_steps_value = args.eval_steps
        else:
            # Convert to integer
            eval_steps_value = int(args.eval_steps)
        
        training_arguments = TrainingArguments(
            output_dir=model_checkpoint_path,
            run_name=f"{experiment_name}_fold_{fold+1}",
            optim='paged_adamw_32bit',  # Changed from paged_adamw_32bit to standard adamw_torch
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            eval_strategy="steps",
            save_strategy='steps',
            logging_steps=8,
            eval_steps=eval_steps_value,
            save_steps=eval_steps_value,
            learning_rate=1e-4,
            fp16=False,
            bf16=False,
            num_train_epochs=args.epochs,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=args.save_total_limit,
            dataloader_drop_last=False,  # To handle small datasets better
        )
        
        # Initialize early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        
        # Create a custom compute_metrics function with access to validation indices and dataset
        def compute_metrics_wrapper(eval_preds):
            return compute_metrics(eval_preds, val_idx, full_dataset)
        
        # Create a custom trainer with class weights for balanced training
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Print debug info for the first batch
                if not hasattr(self, '_printed_debug_info'):
                    print(f"Logits device: {logits.device}, dtype: {logits.dtype}")
                    print(f"Labels device: {labels.device}, dtype: {labels.dtype}")
                    self._printed_debug_info = True
                
                # Move class weights to the same device as logits and ensure consistent dtype
                weights = class_weights_tensor.to(device=logits.device, dtype=torch.float32)
                
                # Convert logits to float32 if necessary
                if logits.dtype != torch.float32:
                    logits = logits.to(torch.float32)
                
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        trainer = Trainer(  # Changed to WeightedTrainer for class weights
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_wrapper,
            callbacks=[early_stopping_callback]
        )
        
        # Print trainable parameters
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
                "eval_accuracy": eval_result["eval_accuracy"],
                "eval_precision": eval_result["eval_precision"],
                "eval_recall": eval_result["eval_recall"],
                "eval_f1": eval_result["eval_f1"]
            }
            
            # Save the model
            model_path = os.path.join(fold_output_dir, "final_model")
            trainer.save_model(model_path)
            
            # Generate predictions for the validation set
            print(f"Generating detailed predictions for fold {fold+1}...")
            
            # Get validation dialogue IDs and examples
            val_data = []
            for i in val_idx:
                val_data.append({
                    "dialogue_id": full_dataset[i]["dialogue_id"],
                    "text": full_dataset[i]["text"],
                    "label": full_dataset[i]["label"],
                    "idx": i
                })
            
            # Run predictions
            fold_predictions = []
            for example in val_data:
                try:
                    # Get the example
                    example_input = tokenizer(
                        example["text"], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=args.max_length,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    # Generate prediction
                    with torch.no_grad():
                        outputs = model(**example_input)
                        logits = outputs.logits
                        probabilities = torch.nn.functional.softmax(logits, dim=-1)
                        predicted_class = torch.argmax(logits, dim=-1).item()
                        confidence = probabilities[0, predicted_class].item()
                    
                    # Add to predictions
                    prediction = {
                        "dialogue_id": example["dialogue_id"],
                        "fold": fold + 1,
                        "text": example["text"],
                        "label": example["label"],
                        "predicted_label": predicted_class,
                        "correct": predicted_class == example["label"],
                        "confidence_score": confidence
                    }
                    
                    fold_predictions.append(prediction)
                except Exception as e:
                    print(f"Error generating prediction for dialogue {example['dialogue_id']}: {e}")
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
            
            # Log to wandb
            try:
                wandb.log({
                    f"fold_{fold+1}/train_loss": fold_result["train_loss"],
                    f"fold_{fold+1}/eval_loss": fold_result["eval_loss"],
                    f"fold_{fold+1}/accuracy": fold_result["eval_accuracy"],
                    f"fold_{fold+1}/precision": fold_result["eval_precision"],
                    f"fold_{fold+1}/recall": fold_result["eval_recall"],
                    f"fold_{fold+1}/f1": fold_result["eval_f1"]
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
        mean_accuracy = np.mean([result["eval_accuracy"] for result in fold_results])
        mean_precision = np.mean([result["eval_precision"] for result in fold_results])
        mean_recall = np.mean([result["eval_recall"] for result in fold_results])
        mean_f1 = np.mean([result["eval_f1"] for result in fold_results])
        
        print(f"Mean Training Loss: {mean_train_loss:.4f}")
        print(f"Mean Evaluation Loss: {mean_eval_loss:.4f}")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")
        print(f"Mean F1 Score: {mean_f1:.4f}")
        
        # Print individual fold results
        for result in fold_results:
            print(f"Fold {result['fold']}: " + 
                 f"Train Loss = {result['train_loss']:.4f}, " +
                 f"Eval Loss = {result['eval_loss']:.4f}, " +
                 f"Accuracy = {result['eval_accuracy']:.4f}, " +
                 f"F1 = {result['eval_f1']:.4f}")
        
        # Log overall metrics to wandb
        try:
            wandb.log({
                "overall/train_loss": mean_train_loss,
                "overall/eval_loss": mean_eval_loss,
                "overall/accuracy": mean_accuracy,
                "overall/precision": mean_precision,
                "overall/recall": mean_recall,
                "overall/f1": mean_f1
            })
        except:
            print("Warning: Could not log to wandb")
    else:
        print("No fold results to report.")
    
    # Calculate overall metrics from saved predictions
    try:
        all_predictions = pd.read_csv(predictions_file)
        
        if len(all_predictions) > 0:
            # Calculate overall accuracy
            overall_accuracy = all_predictions['correct'].mean()
            
            print(f"\nOverall Metrics from Predictions File:")
            print(f"Accuracy = {overall_accuracy:.4f}")
            
            # Log overall metrics to wandb
            try:
                wandb.log({
                    "overall/prediction_accuracy": overall_accuracy
                })
            except:
                print("Warning: Could not log to wandb")
    except Exception as e:
        print(f"Error calculating overall metrics from predictions: {e}")
    
    try:
        wandb.finish()
    except:
        pass
    
    return fold_results


def main():
    args = parse_arguments()
    if args.dataset_type != "p4g":
        return
    
    perform_kfold_cross_validation(args)

if __name__ == "__main__":
    main()