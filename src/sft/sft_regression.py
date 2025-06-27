import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                         BitsAndBytesConfig, TrainingArguments,
                         EarlyStoppingCallback, Trainer)
from trl import SFTTrainer
import re
import math
import gc

class LlamaForRegression(nn.Module):
    """
    Llama model with a regression head for direct numeric prediction.
    """
    def __init__(self, base_model, num_outputs=1, hidden_size=4096):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Simplified regression head to reduce memory
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # Reduced from //2
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_outputs)  # Removed extra layers
        )
        
        # Initialize regression head weights
        self._init_regression_head()
    
    def _init_regression_head(self):
        """Initialize regression head weights"""
        for module in self.regression_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
        else:
            # Fallback for older transformers versions
            self.base_model.config.use_cache = False
            if hasattr(self.base_model, 'enable_input_require_grads'):
                self.base_model.enable_input_require_grads()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
        else:
            # Fallback for older transformers versions
            self.base_model.config.use_cache = True
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs - store hidden states for gradient computation
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Extract last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Pool the sequence dimension (use mean pooling over non-padded tokens)
        if attention_mask is not None:
            # Mask out padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            pooled_output = torch.sum(last_hidden_state * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        else:
            # Simple mean pooling if no attention mask
            pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Pass through regression head
        predictions = self.regression_head(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            
            # FIX: Ensure both predictions and labels have the same shape
            predictions_flat = predictions.squeeze(-1)  # Shape: [batch_size]
            
            # Handle different label shapes properly
            if labels.dim() == 0:  # Scalar tensor
                labels_flat = labels.unsqueeze(0)  # Shape: [1]
            elif labels.dim() == 1:  # 1D tensor
                labels_flat = labels  # Shape: [batch_size]
            else:  # Multi-dimensional tensor
                labels_flat = labels.squeeze()  # Remove extra dimensions
            
            # Ensure both tensors have the same shape
            if predictions_flat.shape != labels_flat.shape:
                # If batch sizes don't match, take the minimum
                min_size = min(predictions_flat.size(0), labels_flat.size(0))
                predictions_flat = predictions_flat[:min_size]
                labels_flat = labels_flat[:min_size]
            
            loss = loss_fct(predictions_flat, labels_flat)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'last_hidden_state': last_hidden_state,
        }
    
    def generate(self, **kwargs):
        """Override generate to prevent accidental text generation"""
        raise NotImplementedError("Regression model does not support text generation. Use forward() for predictions.")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def compute_regression_metrics(eval_preds, dataset_type):
    """
    Compute metrics for regression tasks
    """
    predictions, labels = eval_preds
    predictions = predictions.squeeze()
    
    # Calculate MSE
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate correlation
    correlation = 0.0
    if len(labels) > 1:
        correlation, _ = pearsonr(labels, predictions)
        if np.isnan(correlation):
            correlation = 0.0
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "correlation": correlation
    }
    
    # Dataset-specific metrics
    if dataset_type == "cb":
        # For CB, we care about price prediction accuracy
        metrics["price_mae"] = np.mean(np.abs(labels - predictions))
    
    return metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune Llama 3.1 8B on negotiation datasets with k-fold cross validation")
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["cb", "p4g", "casino"],
                        help="Type of dataset (cb for Craigslist Bargain, p4g for Persuasion for Good)")
    # Intentions arguments
    parser.add_argument("--scaffolding_type", type=str, required=True, choices=["local", "global", "both", "none"],
                        help="Type of intentions to use: local (only intentions), global (only summaries), both, or none")
    parser.add_argument("--summary_type", type=str, default="none",
                        choices=["none", "traditional", "scd", "relational", "scm", "appraisal_theory", "politeness_theory_stage2"],
                        help="Type of summary to use for global intentions")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/data/user_data/rithviks/output/regression",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device (reduced from 4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length (reduced from 1024)")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (reduced from 8)")
    # K-fold arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval_steps", type=int, default=1, help="Steps between evaluations")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of checkpoints to save (reduced)")
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Threshold for early stopping")
    
    return parser.parse_args()

def create_regression_dataset(args, tokenizer, dialogue_info):
    """
    Create dataset specifically for regression training
    """

    regression_data = []
    
    for dialogue_id, info in dialogue_info.items():
        # Get the input prompt
        prompt = info["prompt"]
        
        # Tokenize the prompt
        encoded = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt"
        )
        
        # Get the target value based on dataset type
        target = info["score"]
        
        regression_data.append({
            "dialogue_id": dialogue_id,
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": target
        })

    return regression_data
    

def prepare_dataset(args, tokenizer):
    """Prepare dataset for finetuning with appropriate formatting."""

    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    print("Processing utterance-level dataset...")

    dialogue_info = {}  # Store dialogue info for evaluation
        
    for dialogue_id, group in df.groupby('dialogue_id'):
        # Sort by utterance index
        group = group.sort_values('utterance_idx')
        
        conversation = []
        for _, row in group.iterrows():
            # Format with or without intentions based on scaffolding type
            utterance = row['utterance']

            # Include speaker role context
            speaker = row['speaker']

            if args.scaffolding_type in ["local", "both"] and 'intention' in row and pd.notna(row['intention']):
                # Include intention for local scaffolding
                conversation.append(f"{speaker}: {utterance} [{row['intention']}]")
            else:
                # No intentions
                conversation.append(f"{speaker}: {utterance}")
        
        formatted_conversation = ", ".join(conversation)
        
        # Add summary if using global scaffolding
        summary = ""
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            summary_column = f"{args.summary_type}_summary"
            if summary_column in group.columns and pd.notna(group[summary_column].iloc[0]):
                summary = f", [Summary: {group[summary_column].iloc[0]}]"
        
        # Make sure required columns exist
        required_columns = ['sale_price', 'buyer_target', 'seller_target', 'score']
        if not all(col in group.columns for col in required_columns):
            print(f"Warning: Required columns missing in dialogue {dialogue_id}, skipping")
            continue
            
        buyer_target = group['buyer_target'].iloc[0]
        seller_target = group['seller_target'].iloc[0]
        sale_price = group['sale_price'].iloc[0]
        score = group['score'].iloc[0]
        
        # Always use "with intentions" if they're included
        intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
        
        # Format summary part based on global scaffolding
        summary_part = ""
        if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
            summary_part = f", [summary]"
        
        # Create messages for chat template
        messages = [{
            "role": "user",  
            "content": f"Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict a score between 0 and 1 to signify how close the final projected sale price is to either the buyer target or the seller target. A score of 0 means the projected sale price is equal to the buyer target whereas a score of 1 means the projected sale price is equal to the seller target. Provide only the final answer in the format 'FINAL_SCORE: [score]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}>"
        }]
        
        # Apply chat template - no tokenization, just formatting
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Store dialogue info for evaluation
        dialogue_info[dialogue_id] = {
            "buyer_target": buyer_target,
            "seller_target": seller_target,
            "sale_price": sale_price,
            "score": score,
            "prompt": input_text
        }
    
    # Create regression dataset
    regression_data = create_regression_dataset(args, tokenizer, dialogue_info)
    
    print(f"Prepared regression dataset with {len(regression_data)} examples")
    
    # Show a sample
    if len(regression_data) > 0:
        print("\nSample regression data:")
        sample = regression_data[0]
        print(f"Dialogue ID: {sample['dialogue_id']}")
        print(f"Target: {sample['labels']}")
        print(f"Input length: {len(sample['input_ids'])}")
    
    return regression_data, dialogue_info


def perform_kfold_cross_validation(args):
    """Perform k-fold cross validation."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set environment variable for CUDA memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Clear GPU memory at start
    clear_gpu_memory()
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    regression_data, dialogue_info = prepare_dataset(args, tokenizer)
    
    # Convert to indices for k-fold
    dialogue_ids = [item["dialogue_id"] for item in regression_data]
    indices = list(range(len(regression_data)))
    
    # Initialize KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Track metrics across folds
    fold_results = []
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset_type == "cb":
        results_dir = f"/home/rithviks/FOLIAGE/src/sft/results/craigslistbargainregression/seed_{args.seed}"

    os.makedirs(results_dir, exist_ok=True)
    
    # Extract ratio from filename
    ratio_match = re.search(r'ratio_(\d+\.\d+)', args.dataset_path)
    ratio = ratio_match.group(1) if ratio_match else "unknown"
    
    # Create experiment name for wandb
    experiment_name = f"{args.dataset_type}_ratio_{ratio}_{args.scaffolding_type}"
    if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
        experiment_name += f"_{args.summary_type}"
    
    # Initialize predictions CSV
    predictions_file = os.path.join(results_dir, f"{experiment_name}_regression_predictions.csv")
    
    # Initialize empty DataFrame if file doesn't exist
    if not os.path.exists(predictions_file):
        if args.dataset_type == "cb":
            pd.DataFrame(columns=[
                "dialogue_id", "fold", "prompt", "gold_score", "predicted_score", "buyer_target", 
                "seller_target", "sale_price"
            ]).to_csv(predictions_file, index=False)
    
    # Initialize wandb
    try:
        wandb.init(project=experiment_name, name=experiment_name)
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
    
    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold+1}/{args.n_folds}")
        print(f"{'='*50}")
        
        # Clear GPU memory before each fold
        clear_gpu_memory()
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Split dataset into train and validation
        train_data = [regression_data[i] for i in train_idx]
        val_data = [regression_data[i] for i in val_idx]
        
        print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Initialize model with LoRA for this fold
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # CRITICAL FIX: Apply LoRA BEFORE wrapping with custom class
        base_model = prepare_model_for_kbit_training(base_model)
        
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

        # Apply LoRA to base model
        base_model = get_peft_model(base_model, peft_config)
        
        # CRITICAL FIX: Enable input gradients for the base model
        base_model.enable_input_require_grads()
        
        # Now wrap with regression head
        model = LlamaForRegression(base_model)

        # Set pad token configuration
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        
        # CRITICAL FIX: Ensure regression head parameters require gradients
        for param in model.regression_head.parameters():
            param.requires_grad_(True)

        # Add this right after creating your LlamaForRegression model

        def debug_regression_head(model):
            """Debug regression head parameters"""
            
            print("\n" + "="*30)
            print("DEBUGGING REGRESSION HEAD")
            print("="*30)
            
            if hasattr(model, 'regression_head'):
                print("Regression head found!")
                total_head_params = 0
                trainable_head_params = 0
                
                for i, module in enumerate(model.regression_head):
                    module_params = sum(p.numel() for p in module.parameters())
                    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    
                    print(f"Layer {i} ({type(module).__name__}): {module_params:,} total, {trainable_params:,} trainable")
                    
                    if hasattr(module, 'weight') and module.weight is not None:
                        print(f"  Weight shape: {module.weight.shape}, requires_grad: {module.weight.requires_grad}")
                    if hasattr(module, 'bias') and module.bias is not None:
                        print(f"  Bias shape: {module.bias.shape}, requires_grad: {module.bias.requires_grad}")
                        
                    total_head_params += module_params
                    trainable_head_params += trainable_params
                    
                print(f"\nTotal regression head params: {total_head_params:,}")
                print(f"Trainable regression head params: {trainable_head_params:,}")
                
                # Check if parameters are on the right device
                first_param = next(model.regression_head.parameters())
                print(f"Regression head device: {first_param.device}")
                print(f"Regression head dtype: {first_param.dtype}")
                
            else:
                print("ERROR: No regression head found!")

        # Call this after creating your LlamaForRegression model
        debug_regression_head(model)

        # Set up training arguments with memory optimizations
        model_checkpoint_path = os.path.join(fold_output_dir, "checkpoints")
        
        # Set metric for best model based on dataset type
        if args.dataset_type == "cb":
            metric_for_best_model = "eval_rmse"
            greater_is_better = False
        
        training_arguments = TrainingArguments(
            output_dir=model_checkpoint_path,
            optim='paged_adamw_32bit',
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            log_level='debug',
            eval_strategy="steps",
            save_strategy='steps',
            logging_steps=16,
            eval_steps=0.25,
            save_steps=0.25,
            learning_rate=1e-4,
            fp16=True,
            num_train_epochs=args.epochs,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            lr_scheduler_type='linear',
            save_total_limit=1,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            eval_accumulation_steps=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
        )
        
        # Initialize early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )

        class LlamaRegressionTrainer(Trainer):
            """
            Custom trainer for regression tasks with memory optimizations
            """


            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # FIX: Ensure labels are properly formatted and on correct device
                labels = inputs.get("labels")
                if labels is not None:
                    labels = labels.float()
                    
                    # Ensure labels have the correct shape for batch processing
                    if labels.dim() == 0:  # Scalar
                        labels = labels.unsqueeze(0)  # Make it [1]
                    elif labels.dim() > 1:  # Multi-dimensional
                        labels = labels.view(-1)  # Flatten to 1D
                    
                    # FIX: Move labels to the same device as the model
                    if hasattr(model, 'device'):
                        labels = labels.to(model.device)
                    elif torch.cuda.is_available():
                        labels = labels.cuda()
                    inputs["labels"] = labels
                
                outputs = model(**inputs)
                loss = outputs["loss"]
                
                # CRITICAL FIX: Ensure loss requires gradients
                if loss is not None and not loss.requires_grad:
                    print("Warning: Loss does not require gradients!")
                
                if return_outputs:
                    return loss, {"predictions": outputs.get("predictions")}
                return loss
            
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                labels = inputs.pop("labels") if "labels" in inputs else None
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs["loss"] if labels is not None else None
                    predictions = outputs["predictions"]
                
                if prediction_loss_only:
                    return (loss, None, None)
                
                return (loss, predictions, labels)
        
        # Initialize compute_metrics function
        def compute_metrics_fn(eval_preds):
            return compute_regression_metrics(eval_preds, args.dataset_type)
        
        # Initialize trainer
        trainer = LlamaRegressionTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_arguments,
            compute_metrics=compute_metrics_fn,
            callbacks=[early_stopping_callback],
        )
        
        # Train and evaluate
        try:
            print(f"Starting training for fold {fold+1}...")
            
            # CRITICAL FIX: Verify gradients before training
            print("Checking gradient requirements...")
            base_params_with_grad = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
            head_params_with_grad = sum(p.numel() for p in model.regression_head.parameters() if p.requires_grad)
            print(f"Base model parameters requiring grad: {base_params_with_grad}")
            print(f"Regression head parameters requiring grad: {head_params_with_grad}")
            
            train_result = trainer.train()
            
            print(f"Evaluating fold {fold+1}...")
            eval_result = trainer.evaluate()
            
            # Save results for this fold
            fold_result = {
                "fold": fold + 1,
                "train_loss": train_result.training_loss,
                # "eval_loss": eval_result["eval_loss"],
                "eval_mse": eval_result.get("eval_mse", 0),
                "eval_rmse": eval_result.get("eval_rmse", 0),
                "eval_correlation": eval_result.get("eval_correlation", 0),
            }
            
            # Save the model
            model_path = os.path.join(fold_output_dir, "final_model")
            trainer.save_model(model_path)   

            fold_results.append(fold_result)

            
            # Evaluate on validation set
            print(f"Running detailed evaluation on fold {fold+1}...")
            
            # Get validation dialogue IDs
            val_dialogue_ids = [regression_data[i]["dialogue_id"] for i in val_idx]
            
            # Run predictions
            fold_predictions = []
            for dialogue_id in val_dialogue_ids:
                if dialogue_id not in dialogue_info:
                    continue
                    
                # Get prompt 
                prompt = dialogue_info[dialogue_id]["prompt"]
                
                # Generate prediction
                try:
                    encoded = tokenizer(
                        prompt,
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_length,
                        return_tensors="pt"
                    )
                    
                    # FIX: Move all tensors to the same device as the model
                    device = next(model.parameters()).device
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    labels = torch.tensor([dialogue_info[dialogue_id]["score"]], dtype=torch.float).to(device)
                    
                    regression_input = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                    
                    with torch.no_grad():
                        outputs = model(**regression_input)
                        predicted_score = outputs["predictions"].cpu().numpy()[0][0]
    
                    # Add to predictions
                    prediction = {
                        "dialogue_id": dialogue_id,
                        "fold": fold + 1,
                        "prompt": prompt,
                        "buyer_target": dialogue_info[dialogue_id]["buyer_target"],
                        "seller_target": dialogue_info[dialogue_id]["seller_target"],
                        "sale_price": dialogue_info[dialogue_id]["sale_price"],
                        "gold_score": dialogue_info[dialogue_id]["score"],
                        "predicted_score": predicted_score
                    }
                        
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
            
            # Clear memory after each fold
            del model, trainer, base_model
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error during training or evaluation for fold {fold+1}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up memory even on error
            try:
                del model, trainer, base_model
            except:
                pass
            clear_gpu_memory()
            continue
    
    # Print cross-validation summary
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    
    if fold_results:
        # Calculate aggregate metrics
        mean_train_loss = np.mean([result["train_loss"] for result in fold_results])
        # mean_eval_loss = np.mean([result["eval_loss"] for result in fold_results])
        mean_rmse = np.mean([result["eval_rmse"] for result in fold_results])
        mean_correlation = np.mean([result["eval_correlation"] for result in fold_results])
        
        print(f"Mean Training Loss: {mean_train_loss:.4f}")
        # print(f"Mean Evaluation Loss: {mean_eval_loss:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f}")
        print(f"Mean Correlation: {mean_correlation:.4f}")
    else:
        print("No fold results to report.")
    
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