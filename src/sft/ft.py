import os
import torch
import pandas as pd
import argparse
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune Llama 3.1 8B on negotiation datasets with local or global intentions")
    
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
    parser.add_argument("--output_dir", type=str, default="/data/user_data/gganeshl/", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, 
                        help="Logging steps")
    parser.add_argument("--lora_rank", type=int, default=8, 
                        help="LoRA rank parameter")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length")

    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push the model to the Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for uploading to the Hugging Face Hub (e.g., 'username/model-name')")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="Hugging Face Hub token for uploading the model")
    
    return parser.parse_args()

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
                    summary = f"\n\nSummary: {group[summary_column].iloc[0]}"
            
            # Get outcome/label
            if args.dataset_type == "cb":
                outcome = f"Sale Price: ${group['sale_price'].iloc[0]}"
                buyer_target = group['buyer_target'].iloc[0]
                seller_target = group['seller_target'].iloc[0]
                context = f"Buyer Target: ${buyer_target}, Seller Target: ${seller_target}\n\n"
            elif args.dataset_type == "p4g":
                outcome = f"Donation Made: {'Yes' if group['label'].iloc[0].lower() == 'yes' else 'No'}"
                context = ""
            
            # Format prompt
            if args.dataset_type == "cb":
                input_text = f"<s>[INST] You are helping analyze a negotiation conversation.\n\n{context}Conversation:\n{formatted_conversation}{summary}\n\nPredict the final sale price. [/INST]"
            else:  # p4g
                input_text = f"<s>[INST] You are helping analyze a persuasion conversation.\n\n{context}Conversation:\n{formatted_conversation}{summary}\n\nPredict whether the persuadee will make a donation. [/INST]"
            
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
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Show a sample
    print("\nSample input-output pair:")
    sample_idx = 0
    print(f"INPUT:\n{dataset[sample_idx]['input']}")
    print(f"OUTPUT:\n{dataset[sample_idx]['output']}")
    
    return dataset

def setup_model_and_tokenizer():
    """Set up the Llama 3.1 8B model with 4-bit quantization."""
    print("Setting up model and tokenizer...")
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    
    try:
        # First attempt with 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        print("Successfully loaded model with 4-bit quantization")
        
    except Exception as e:
        print(f"Error with 4-bit quantization: {str(e)}")
        print("Falling back to 8-bit quantization...")
        
        try:
            # Try 8-bit quantization as fallback
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            model = prepare_model_for_kbit_training(model)
            print("Successfully loaded model with 8-bit quantization")
            
        except Exception as e2:
            print(f"Error with 8-bit quantization: {str(e2)}")
            print("Falling back to 16-bit (no quantization)...")
            
            # Final fallback to 16-bit
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("Successfully loaded model in 16-bit (no quantization)")
    
    return model, tokenizer

def train_model(args, model, tokenizer, dataset):
    """Set up and run the finetuning process."""
    print("Setting up training configuration...")
    
    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        bf16=True,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )
    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        max_seq_length=args.max_length,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    model_name = f"llama-3.1-8b-{args.dataset_type}-{args.scaffolding_type}"
    if args.scaffolding_type in ["global", "both"]:
        model_name += f"-{args.summary_type}"
    
    output_dir = os.path.join(args.output_dir, model_name)
    
    # Save model locally
    trainer.save_model(output_dir)
    print(f"Model saved locally to {output_dir}")
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        if args.hub_model_id is None:
            # Generate a hub model ID if not provided
            hub_model_id = model_name
            print(f"No hub_model_id provided, using '{hub_model_id}'")
        else:
            hub_model_id = args.hub_model_id
            
        print(f"Pushing model to Hugging Face Hub as {hub_model_id}...")
        
        # If we already saved locally, we push from the saved model
        if not trainer.args.should_save:
            # This will upload the model directly
            trainer.model.push_to_hub(
                hub_model_id, 
                token=args.hub_token,
                commit_message=f"Finetuned Llama 3.1 8B on {args.dataset_type} with {args.scaffolding_type} scaffolding"
            )
            print(f"Model pushed to Hub: https://huggingface.co/{hub_model_id}")
    
    return output_dir

def main():
    args = parse_arguments()
    
    print("=" * 50)
    print(f"Finetuning Llama 3.1 8B on {args.dataset_type} dataset")
    print(f"Scaffolding type: {args.scaffolding_type}")
    if args.scaffolding_type in ["global", "both"]:
        print(f"Summary type: {args.summary_type}")
    print("=" * 50)
    
    # Prepare dataset
    dataset = prepare_dataset(args)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Train model
    output_dir = train_model(args, model, tokenizer, dataset)
    
    print("=" * 50)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()