import unsloth
import os
import torch
import pandas as pd
import argparse
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel  # Import Unsloth's FastLanguageModel

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
    parser.add_argument("--lora_rank", type=int, default=16,  # Updated default to 16 for Unsloth
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
                    summary = f", [Summary: {group[summary_column].iloc[0]}]"
            
            # Get outcome/label
            if args.dataset_type == "cb":
                outcome = f"FINAL_PRICE: ${group['sale_price'].iloc[0]}"
                buyer_target = group['buyer_target'].iloc[0]
                seller_target = group['seller_target'].iloc[0]
                
                # Always use "with intentions" if they're included
                intentions_note = " with intentions" if args.scaffolding_type in ["local", "both"] else ""
                
                # Format summary part based on global scaffolding
                summary_part = ""
                if args.scaffolding_type in ["global", "both"] and args.summary_type != "none":
                    summary_part = f", [summary]"
                
                # Use the exact prompt template
                input_text = f"<s>[INST] Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{summary_part}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: [number]'\nINPUT: <${buyer_target}, ${seller_target}, [{formatted_conversation}]{summary}> [/INST]"
                
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
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Show a sample
    print("\nSample input-output pair:")
    sample_idx = 0
    print(f"INPUT:\n{dataset[sample_idx]['input']}")
    print(f"OUTPUT:\n{dataset[sample_idx]['output']}")
    
    return dataset

def setup_model_and_tokenizer(args):
    """Set up the Llama 3.1 8B model with Unsloth optimizations."""
    print("Setting up model and tokenizer with Unsloth...")
    
    # Determine dtype based on device capabilities
    dtype = torch.float16
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    
    # Using Unsloth to load the model in 4-bit
    try:
        # Try with Meta-Llama-3.1-8B first
        model_name = "meta-llama/Meta-Llama-3.1-8B"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.max_length,
            dtype=dtype,
            load_in_4bit=True,
            # token="your_hf_token"  # Uncomment if needed for gated models
        )
        print(f"Successfully loaded {model_name} with Unsloth")
    except Exception as e:
        print(f"Error loading Meta-Llama-3.1-8B: {str(e)}")
        print("Falling back to Llama-3-8B...")
        
        try:
            # Try with unsloth's Llama-3-8B version
            model_name = "unsloth/llama-3-8b-bnb-4bit"
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=args.max_length,
                dtype=dtype,
                load_in_4bit=True,
            )
            print(f"Successfully loaded {model_name} with Unsloth")
        except Exception as e2:
            print(f"Error loading Llama-3-8B: {str(e2)}")
            raise RuntimeError("Failed to load model with Unsloth. Please check your installation and token.")
    
    # Set up LoRA with Unsloth optimizations
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,  # Typically set to same value as r
        lora_dropout=0,             # 0 is optimized in Unsloth
        bias="none",                # "none" is optimized in Unsloth
        use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
        random_state=3407,
        use_rslora=False,           # Not using rank stabilized LoRA
    )
    
    return model, tokenizer

def train_model(args, model, tokenizer, dataset):
    """Set up and run the finetuning process with Unsloth optimizations."""
    print("Setting up training configuration...")
    
    # Max steps calculation (prefer steps over epochs for consistency)
    steps_per_epoch = len(dataset) // (args.batch_size * 4)  # Assuming gradient_accumulation_steps=4
    max_steps = steps_per_epoch * args.epochs
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_8bit",  # Unsloth recommends this
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=steps_per_epoch,  # Save once per epoch
        warmup_steps=int(0.05 * max_steps),  # 5% warmup
        lr_scheduler_type="linear",
        report_to="none",
        seed=3407,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )
    
    # Set up trainer with Unsloth-optimized settings
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        dataset_num_proc=2,
        packing=False,  # Unsloth docs suggest this can make training 5x faster for short sequences
    )
    
    # Start training
    print("Starting training with Unsloth optimizations...")
    trainer_stats = trainer.train()
    
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
    
    print("Training stats:")
    print(f"Total training time: {trainer_stats.metrics.get('total_training_time', 'N/A')}")
    print(f"Training loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    print(f"Training steps per second: {trainer_stats.metrics.get('train_steps_per_second', 'N/A')}")
    
    return output_dir

def main():
    args = parse_arguments()
    
    print("=" * 50)
    print(f"Finetuning Llama 3.1 8B on {args.dataset_type} dataset using Unsloth")
    print(f"Scaffolding type: {args.scaffolding_type}")
    if args.scaffolding_type in ["global", "both"]:
        print(f"Summary type: {args.summary_type}")
    print("=" * 50)
    
    # Prepare dataset
    dataset = prepare_dataset(args)
    
    # Setup model and tokenizer with Unsloth
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Train model
    output_dir = train_model(args, model, tokenizer, dataset)
    
    print("=" * 50)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()