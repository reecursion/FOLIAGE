import numpy as np
import math
import argparse
import json
import pandas as pd
from pprint import pprint
import os
from tqdm import tqdm
import wandb
from shap_analysis import run_shap_analysis
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from dataloader import *
from models import *
import transformers


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed
)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='p4g', help='Choosing the CONV Forecasting dataset')
    parser.add_argument('--input_dir',      type=str, default='baselines/data/', help='The input directory')
    parser.add_argument('--local_info',     type=str, default=None, help='Choice of local_information to use')
    parser.add_argument('--global_info',    type=str, default=None, help='Choice of global_information to use')
    parser.add_argument('--LLM',            type=str, default='gpt-4o', help='The language model to use')
    parser.add_argument('--frac',           type=str, default='ALL', help='The fraction of data to use')
    parser.add_argument('--fold',           type=int, default=0, help='The fold to use')

    parser.add_argument('--model_name',     type=str, default='bert-base-uncased', help='The model to use') # backbone model to use

    parser.add_argument('--trainable',      type=bool, default=True, help='Whether to train the model')
    parser.add_argument('--do_train',       type=int, default=1, help='Whether to train the model')
    parser.add_argument('--do_test',        type=int, default=1, help='Whether to test the model')
    parser.add_argument('--max_seq_len',    type=int, default=512, help='The maximum sequence length')
    parser.add_argument('--batch_size',     type=int, default=4, help='The batch size')
    parser.add_argument('--gpu',            type=str, default='0', help='The gpu to use')

    parser.add_argument('--epochs',         type=int, default=15, help='The number of training epochs')
    parser.add_argument('--seed',           type=int, default=42, help='The random seed')
    parser.add_argument('--learning_rate',  type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='The number of gradient accumulation steps')
    parser.add_argument('--patience',       type=int, default=5, help='The number of patience steps')
    
    # Multi-task learning arguments
    parser.add_argument('--task1_classes',  type=int, default=2, help='Number of classes for task 1 (binary_score)')
    parser.add_argument('--task2_classes',  type=int, default=3, help='Number of classes for task 2 (new_label)')
    parser.add_argument('--task1_weight',   type=float, default=1.0, help='Weight for primary task (binary_score)')
    parser.add_argument('--task2_weight',   type=float, default=1.0, help='Weight for secondary task (new_label)')
    parser.add_argument('--combine_metric', type=str, default='weighted_avg', choices=['weighted_avg', 'task1_only', 'task2_only'], 
                        help='How to combine metrics for early stopping')
    
    parser.add_argument('--run_shap',           type=int, default=0, help='Whether to run SHAP analysis')
    parser.add_argument('--shap_samples',       type=int, default=100, help='Number of samples for SHAP analysis')
    parser.add_argument('--shap_background',    type=int, default=50, help='Number of background samples for SHAP')
    parser.add_argument('--category',           type=int, default=1, help='Category for pairwise classifier')

    args = parser.parse_args()
    return args


def seen_eval(model, data_loader, device, args, tokenizer, save_csv=False, csv_path=None):
    model.eval()
    
    # Separate predictions for each task
    preds_task1, targets_task1 = [], []
    preds_task2, targets_task2 = [], []
    
    total_loss = 0.0
    total_task1_loss = 0.0
    total_task2_loss = 0.0
    num_batches = 0
    per_summary_rows = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            try:
                upd_data = {
                    "utt_ids": [elem.to(device) for elem in data["utt_ids"]],
                    "utt_masks": [elem.to(device) for elem in data["utt_masks"]]
                }
                if data["global_ids"] is not None:
                    upd_data["global_ids"] = data["global_ids"].to(device)
                    upd_data["global_masks"] = data["global_masks"].to(device)

                targets_1 = data["binary_score"].to(device)
                targets_2 = data["new_label"].to(device)
                
                output_dict = model(upd_data)
                logits_task1 = output_dict['logits_task1']
                logits_task2 = output_dict['logits_task2']
                
                probs_task1 = F.softmax(logits_task1, dim=1)
                probs_task2 = F.softmax(logits_task2, dim=1)

                # Compute losses for each task
                loss_task1 = CE_loss(logits_task1, targets_1)
                loss_task2 = CE_loss(logits_task2, targets_2)
                
                # Combined loss
                total_loss_batch = args.task1_weight * loss_task1 + args.task2_weight * loss_task2

                total_loss += total_loss_batch.item()
                total_task1_loss += loss_task1.item()
                total_task2_loss += loss_task2.item()
                
                pred_labels_task1 = torch.argmax(logits_task1, dim=1)
                pred_labels_task2 = torch.argmax(logits_task2, dim=1)

                preds_task1 += pred_labels_task1.tolist()
                targets_task1 += targets_1.tolist()
                preds_task2 += pred_labels_task2.tolist()
                targets_task2 += targets_2.tolist()
                
                num_batches += 1

                if save_csv:
                    for j in range(len(targets_1)):
                        row = {
                            "dialogue_id": data.get("dialogue_id", [None])[j],
                            "utterance": ' '.join(data.get("utt_text", [None])[j]),
                            "gold_label_task1": targets_1[j].item(),
                            "predicted_label_task1": pred_labels_task1[j].item(),
                            "confidence_task1": probs_task1[j][pred_labels_task1[j].item()].item(),
                            "correct_task1": pred_labels_task1[j].item() == targets_1[j].item(),
                            "gold_label_task2": targets_2[j].item(),
                            "predicted_label_task2": pred_labels_task2[j].item(),
                            "confidence_task2": probs_task2[j][pred_labels_task2[j].item()].item(),
                            "correct_task2": pred_labels_task2[j].item() == targets_2[j].item()
                        }
                        per_summary_rows.append(row)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[OOM] Skipping eval batch {i}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    # Calculate metrics for each task
    avg_loss = total_loss / max(1, num_batches)
    avg_task1_loss = total_task1_loss / max(1, num_batches)
    avg_task2_loss = total_task2_loss / max(1, num_batches)
    
    # Task 1 metrics
    p1, r1, f1_1, _ = precision_recall_fscore_support(targets_task1, preds_task1, average="macro", zero_division=0)
    
    # Task 2 metrics  
    p2, r2, f1_2, _ = precision_recall_fscore_support(targets_task2, preds_task2, average="macro", zero_division=0)
    
    # Combined metric for early stopping
    if args.combine_metric == 'weighted_avg':
        combined_f1 = (args.task1_weight * f1_1 + args.task2_weight * f1_2) / (args.task1_weight + args.task2_weight)
    elif args.combine_metric == 'task1_only':
        combined_f1 = f1_1
    elif args.combine_metric == 'task2_only':
        combined_f1 = f1_2

    # Save predictions to CSV
    if save_csv and csv_path:
        df = pd.DataFrame(per_summary_rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved per-summary results to: {csv_path}")

    return {
        "task1_precision": p1, "task1_recall": r1, "task1_f1": f1_1, "task1_loss": avg_task1_loss,
        "task2_precision": p2, "task2_recall": r2, "task2_f1": f1_2, "task2_loss": avg_task2_loss,
        "combined_f1": combined_f1, "total_loss": avg_loss
    }


if __name__ == '__main__':

    args = get_arguments()
    pprint(args)

    if args.dataset == 'cd' or args.dataset == 'p4g':
        args.num_classes = 2  # Keep this for backward compatibility, but we'll use task-specific classes

    seed_everything(args.seed)

    # Load the data 
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = json.load(open(f'baselines/data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_train.json'))
    dev_data = json.load(open(f'baselines/data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_test.json'))
    test_data = json.load(open(f'baselines/data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_test.json'))

    # Check if new_label exists in the data
    print("============== DATA INSPECTION ==============")
    sample_item = train_data[0]
    print("Available keys in first training item:")
    print(list(sample_item.keys()))
    
    if 'new_label' not in sample_item:
        print("ERROR: 'new_label' field not found in data!")
        print("Please add 'new_label' to your JSON data files.")
        print("For now, creating dummy new_label values...")
        
        # Create dummy labels for testing (remove this in production)
        import random
        for item in train_data:
            item['new_label'] = random.randint(0, 1)
        for item in dev_data:
            item['new_label'] = random.randint(0, 1)
        for item in test_data:
            item['new_label'] = random.randint(0, 1)
        print("Added dummy new_label values (0 or 1)")
    
    print(f"Sample item keys: {list(train_data[0].keys())}")
    print(f"binary_score: {train_data[0]['binary_score']}")
    print(f"new_label: {train_data[0]['new_label']}")

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_data, dev_data, test_data, tokenizer, args,
    )

    model = BERT_HierarchicalTransformer(args)
    model.to(device)
    CE_loss = nn.CrossEntropyLoss()

    identifiable_file = f'{args.dataset}_multitask_classification_ratio_{args.frac}_{args.local_info}_{args.global_info}_{args.model_name}_fold_{args.fold}'
    checkpoint_file = f'/data/shire/projects/RAT_forecast/ckpts/{identifiable_file}.pt'


    if args.do_train == 1:
        
        # Validate data before training
        print("============== VALIDATING DATA ==============")
        print("Checking training data labels...")
        
        all_binary_scores = [item['binary_score'] for item in train_data]
        all_new_labels = [item['new_label'] for item in train_data]
        
        print(f"binary_score range: [{min(all_binary_scores)}, {max(all_binary_scores)}]")
        print(f"new_label range: [{min(all_new_labels)}, {max(all_new_labels)}]")
        print(f"binary_score unique values: {sorted(set(all_binary_scores))}")
        print(f"new_label unique values: {sorted(set(all_new_labels))}")
        
        # Check for invalid values
        invalid_binary = [x for x in all_binary_scores if x < 0 or x >= args.task1_classes]
        invalid_new = [x for x in all_new_labels if x < 0 or x >= args.task2_classes]
        
        if invalid_binary:
            print(f"ERROR: Found invalid binary_score values: {set(invalid_binary)}")
            print(f"Expected range: [0, {args.task1_classes-1}]")
            exit(1)
            
        if invalid_new:
            print(f"ERROR: Found invalid new_label values: {set(invalid_new)}")
            print(f"Expected range: [0, {args.task2_classes-1}]")
            exit(1)
            
        print("Data validation passed!")

        #######################################
        # TRAINING LOOP                       #
        #######################################
        print("Setting up training loop...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best_combined_f1 = 0
        best_task1_metrics = {}
        best_task2_metrics = {}
        kill_cnt = 0

        for epoch in range(args.epochs):
            print(f"============== TRAINING ON EPOCH {epoch} ==============")
            running_loss = 0.0
            running_task1_loss = 0.0
            running_task2_loss = 0.0
            model.train()

            for i, data in enumerate(tqdm(train_loader)):
                try:
                    # load the data from the data loader
                    upd_data = {
                        "utt_ids": [elem.to(device) for elem in data["utt_ids"]],
                        "utt_masks": [elem.to(device) for elem in data["utt_masks"]]
                    }
                    if data["global_ids"] is not None:
                        upd_data["global_ids"] = data["global_ids"].to(device)
                        upd_data["global_masks"] = data["global_masks"].to(device)

                    targets_1 = data["binary_score"].to(device)
                    targets_2 = data["new_label"].to(device)
                    
                    # Validate target labels
                    if torch.any(targets_1 < 0) or torch.any(targets_1 >= args.task1_classes):
                        print(f"Invalid targets_1 at batch {i}: {targets_1}")
                        print(f"Min: {torch.min(targets_1)}, Max: {torch.max(targets_1)}")
                        continue
                    
                    if torch.any(targets_2 < 0) or torch.any(targets_2 >= args.task2_classes):
                        print(f"Invalid targets_2 at batch {i}: {targets_2}")
                        print(f"Min: {torch.min(targets_2)}, Max: {torch.max(targets_2)}")
                        continue
                    
                    # forward pass
                    optimizer.zero_grad()
                    output_dict = model(upd_data)

                    # compute losses for both tasks
                    logits_task1 = output_dict['logits_task1']
                    logits_task2 = output_dict['logits_task2']
                    
                    # Validate logits shapes
                    if logits_task1.size(0) != targets_1.size(0):
                        print(f"Batch size mismatch - logits_task1: {logits_task1.shape}, targets_1: {targets_1.shape}")
                        continue
                    
                    if logits_task2.size(0) != targets_2.size(0):
                        print(f"Batch size mismatch - logits_task2: {logits_task2.shape}, targets_2: {targets_2.shape}")
                        continue

                    loss_task1 = CE_loss(logits_task1, targets_1)
                    loss_task2 = CE_loss(logits_task2, targets_2)
                    
                    # Combined loss with weights
                    total_loss = args.task1_weight * loss_task1 + args.task2_weight * loss_task2

                    total_loss.backward()
                    running_loss += total_loss.item()
                    running_task1_loss += loss_task1.item()
                    running_task2_loss += loss_task2.item()
                    
                    if (i + 1) % args.grad_accumulation_steps == 0:
                        optimizer.step()
                        
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        print(f"CUDA error at batch {i}:")
                        print(f"targets_1: {data['binary_score']}")
                        print(f"targets_2: {data['new_label']}")
                        print(f"targets_1 range: [{torch.min(data['binary_score'])}, {torch.max(data['binary_score'])}]")
                        print(f"targets_2 range: [{torch.min(data['new_label'])}, {torch.max(data['new_label'])}]")
                        print(f"Error: {e}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            # final gradient step if we hadn't handled it already
            if (i + 1) % args.grad_accumulation_steps != 0:
                optimizer.step()

            print("============== EVALUATION ==============")

            # Training evaluation
            results = seen_eval(model, train_loader, device, args, tokenizer)
            print(f"Train Task1 \t P: {results['task1_precision']:.4f} \t R: {results['task1_recall']:.4f} \t F1: {results['task1_f1']:.4f}")
            print(f"Train Task2 \t P: {results['task2_precision']:.4f} \t R: {results['task2_recall']:.4f} \t F1: {results['task2_f1']:.4f}")
            print(f"Train Combined F1: {results['combined_f1']:.4f} \t Total Loss: {results['total_loss']:.4f}")

            # Validation evaluation
            results = seen_eval(model, dev_loader, device, args, tokenizer)
            print(f"Val Task1 \t P: {results['task1_precision']:.4f} \t R: {results['task1_recall']:.4f} \t F1: {results['task1_f1']:.4f}")
            print(f"Val Task2 \t P: {results['task2_precision']:.4f} \t R: {results['task2_recall']:.4f} \t F1: {results['task2_f1']:.4f}")
            print(f"Val Combined F1: {results['combined_f1']:.4f} \t Total Loss: {results['total_loss']:.4f}")

            # Early stopping based on combined metric
            if results['combined_f1'] > best_combined_f1:
                kill_cnt = 0
                best_combined_f1 = results['combined_f1']
                best_task1_metrics = {
                    'precision': results['task1_precision'],
                    'recall': results['task1_recall'],
                    'f1': results['task1_f1']
                }
                best_task2_metrics = {
                    'precision': results['task2_precision'],
                    'recall': results['task2_recall'],
                    'f1': results['task2_f1']
                }
                torch.save(model.state_dict(), checkpoint_file)
            else:
                kill_cnt += 1
                if kill_cnt >= args.patience:
                    break
            
            print(f"[best val] Combined F1: {best_combined_f1:.4f}")
            print(f"[best val] Task1 - P: {best_task1_metrics['precision']:.4f}, R: {best_task1_metrics['recall']:.4f}, F1: {best_task1_metrics['f1']:.4f}")
            print(f"[best val] Task2 - P: {best_task2_metrics['precision']:.4f}, R: {best_task2_metrics['recall']:.4f}, F1: {best_task2_metrics['f1']:.4f}")

        print("============== EVALUATION ON TEST DATA ==============")
        model.load_state_dict(torch.load(checkpoint_file))
        model.to(device)
        model.eval()

        csv_path = f"baselines/results/{args.dataset}/seed_{args.seed}/{identifiable_file}_test.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        results = seen_eval(
            model, 
            test_loader, 
            device, 
            args, 
            tokenizer, 
            save_csv=True, 
            csv_path=csv_path
        )

        print(f"Test Task1 \t P: {results['task1_precision']:.4f} \t R: {results['task1_recall']:.4f} \t F1: {results['task1_f1']:.4f}")
        print(f"Test Task2 \t P: {results['task2_precision']:.4f} \t R: {results['task2_recall']:.4f} \t F1: {results['task2_f1']:.4f}")
        print(f"Test Combined F1: {results['combined_f1']:.4f} \t Total Loss: {results['total_loss']:.4f}")

    # SHAP Analysis remains the same as before
    if args.run_shap == 1:
        print("============== STARTING SHAP ANALYSIS ==============")
        
        # Load the best model
        if os.path.exists(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file))
            model.to(device)
            model.eval()
            print(f"Loaded model from {checkpoint_file}")
        else:
            print("No checkpoint found, using current model state")
        
        # Prepare data for SHAP analysis
        test_samples = test_data[:args.shap_samples]
        background_samples = train_data[:args.shap_background]
        
        # Run SHAP analysis
        try:
            shap_results = run_shap_analysis(
                model=model,
                tokenizer=tokenizer,
                device=device,
                args=args,
                test_data=test_samples,
                background_data=background_samples
            )
            
            # Save SHAP results
            shap_results_dir = f"baselines/shap_results/{args.dataset}/seed_{args.seed}"
            os.makedirs(shap_results_dir, exist_ok=True)
            
            # Save intention impact results
            intention_impact = shap_results['intention_results']['intention_impact']
            np.save(f"{shap_results_dir}/intention_impact_{identifiable_file}.npy", intention_impact)
            
            # Save summary of results
            with open(f"{shap_results_dir}/shap_summary_{identifiable_file}.txt", 'w') as f:
                f.write(f"SHAP Analysis Results for {args.dataset}\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Local Info: {args.local_info}\n")
                f.write(f"Global Info: {args.global_info}\n")
                f.write(f"Samples Analyzed: {len(test_samples)}\n")
                f.write(f"Background Samples: {len(background_samples)}\n\n")
                
                # Intention impact summary
                mean_intention_impact = np.mean(intention_impact[:, 1])  # Positive class impact
                f.write(f"Mean Intention Impact on Positive Class: {mean_intention_impact:.4f}\n")
                f.write(f"Std Intention Impact: {np.std(intention_impact[:, 1]):.4f}\n")
                
                # Summary impact results
                if 'summary_results' in shap_results:
                    f.write(f"\nGlobal Summary Impact Results:\n")
                    summary_types = ['traditional_summary', 'relational_summary', 'scm_summary', 
                                   'scd_summary', 'politeness_summary']
                    
                    for summary_type in summary_types:
                        if summary_type in shap_results['summary_results'] and \
                           'impact' in shap_results['summary_results'][summary_type]:
                            impact = shap_results['summary_results'][summary_type]['impact'][:, 1]
                            mean_impact = np.mean(impact)
                            f.write(f"{summary_type}: {mean_impact:.4f}\n")
            
            print(f"SHAP analysis completed. Results saved to {shap_results_dir}")
            
            # Print key findings
            print("\n============== SHAP ANALYSIS SUMMARY ==============")
            print(f"Mean Intention Impact: {mean_intention_impact:.4f}")
            
            if 'summary_results' in shap_results:
                print("\nGlobal Summary Impacts:")
                for summary_type in summary_types:
                    if summary_type in shap_results['summary_results'] and \
                       'impact' in shap_results['summary_results'][summary_type]:
                        impact = shap_results['summary_results'][summary_type]['impact'][:, 1]
                        mean_impact = np.mean(impact)
                        print(f"  {summary_type}: {mean_impact:.4f}")
            
        except Exception as e:
            print(f"Error during SHAP analysis: {str(e)}")
            import traceback
            traceback.print_exc()