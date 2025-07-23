import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
import os
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
import itertools

# Import your existing modules
from dataloader import get_data_loaders
from models import BERT_HierarchicalTransformer
from transformers import AutoTokenizer

def get_pairwise_mappings():
    """Define the 15 pairwise classifications and their corresponding class mappings"""
    
    # Define the 6 classes
    classes = [
        'expressed_donate_did',      # 0: Expressed intention of donating but did not donate
        'expressed_donate_donated',  # 1: Expressed intention and donated  
        'no_express_donated',        # 2: Did not express intention but donated
        'no_express_no_donate',      # 3: Did not express intention and did not donate
        'unclear_donated',           # 4: Unclear but donated
        'unclear_no_donate'          # 5: Unclear but did not donate
    ]
    
    # Generate all possible pairs (15 total)
    pairwise_mappings = []
    pair_id = 1
    
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            class1, class2 = classes[i], classes[j]
            pairwise_mappings.append({
                'pair_id': f"pairwise_{pair_id:02d}",
                'class1': class1,
                'class2': class2,
                'class1_idx': i,
                'class2_idx': j
            })
            pair_id += 1
    
    return pairwise_mappings, classes

def load_pairwise_model(pair_info, fold, args_template):
    """Load a specific pairwise classifier model"""
    
    # Create args for this specific pair
    args = argparse.Namespace(**vars(args_template))
    args.category = pair_info['pair_id'].split('_')[1]  # Extract pair number
    if args.category[0] == '0':
        args.category = args.category[1:]
    args.fold = fold
    args.num_classes = 2  # Pairwise classification
    
    # Initialize model
    model = BERT_HierarchicalTransformer(args)
    
    # Construct checkpoint filename based on your naming convention
    identifiable_file = f'{args.dataset}_classification_ratio_{args.frac}_{args.local_info}_{args.global_info}_{args.model_name}_fold_{args.fold}_cat{args.category}'
    checkpoint_file = f'/data/shire/projects/RAT_forecast/ckpts/{identifiable_file}.pt'
    
    if os.path.exists(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file, map_location=args.device))
        model.to(args.device)
        model.eval()
        print(f"Loaded model: {checkpoint_file}")
        return model
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_file}")
        return None

def predict_pairwise(model, data_loader, device, pair_info):
    """Get predictions from a pairwise classifier"""
    model.eval()
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Predicting {pair_info['pair_id']}"):
            try:
                upd_data = {
                    "utt_ids": [elem.to(device) for elem in data["utt_ids"]],
                    "utt_masks": [elem.to(device) for elem in data["utt_masks"]]
                }
                if data["global_ids"] is not None:
                    upd_data["global_ids"] = data["global_ids"].to(device)
                    upd_data["global_masks"] = data["global_masks"].to(device)

                output_dict = model(upd_data)
                logits = output_dict['logits']
                probs = F.softmax(logits, dim=1)
                
                pred_labels = torch.argmax(logits, dim=1)
                batch_confidences = torch.max(probs, dim=1)[0]
                
                predictions.extend(pred_labels.cpu().tolist())
                confidences.extend(batch_confidences.cpu().tolist())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[OOM] Skipping batch in {pair_info['pair_id']}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    return predictions, confidences

def majority_vote_classification(pairwise_results, pairwise_mappings, num_classes=6):
    """
    Perform majority vote classification from pairwise results
    
    Args:
        pairwise_results: List of dictionaries containing pairwise predictions
        pairwise_mappings: List of pairwise mapping information
        num_classes: Number of total classes (6)
    
    Returns:
        final_predictions: List of final class predictions
        confidence_scores: List of confidence scores for final predictions
    """
    
    num_samples = len(pairwise_results[0]['predictions'])
    final_predictions = []
    confidence_scores = []
    
    for sample_idx in range(num_samples):
        # Count votes for each class
        class_votes = [0] * num_classes
        class_confidences = [[] for _ in range(num_classes)]
        
        for pair_idx, pair_result in enumerate(pairwise_results):
            pair_info = pairwise_mappings[pair_idx]
            prediction = pair_result['predictions'][sample_idx]
            confidence = pair_result['confidences'][sample_idx]
            
            # Map pairwise prediction to original class
            if prediction == 0:  # First class in the pair
                voted_class = pair_info['class1_idx']
            else:  # Second class in the pair
                voted_class = pair_info['class2_idx']
            
            class_votes[voted_class] += 1
            class_confidences[voted_class].append(confidence)
        
        # Find the class with maximum votes
        max_votes = max(class_votes)
        winning_classes = [i for i, votes in enumerate(class_votes) if votes == max_votes]
        
        if len(winning_classes) == 1:
            final_class = winning_classes[0]
        else:
            # Break ties using average confidence
            tie_confidences = []
            for cls in winning_classes:
                if class_confidences[cls]:
                    avg_conf = np.mean(class_confidences[cls])
                else:
                    avg_conf = 0.0
                tie_confidences.append(avg_conf)
            
            best_tie_idx = np.argmax(tie_confidences)
            final_class = winning_classes[best_tie_idx]
        
        # Calculate confidence as the average confidence of votes for the winning class
        if class_confidences[final_class]:
            final_confidence = np.mean(class_confidences[final_class])
        else:
            final_confidence = 1.0 / num_classes  # Default confidence
        
        final_predictions.append(final_class)
        confidence_scores.append(final_confidence)
    
    return final_predictions, confidence_scores

def aggregate_across_folds(all_fold_results, class_names):
    """Aggregate results across all folds to get final predictions"""
    
    # Get all dialogue IDs
    all_dialogue_ids = set()
    for fold_results in all_fold_results.values():
        all_dialogue_ids.update(fold_results['dialogue_ids'])
    
    final_results = {}
    
    for dialogue_id in all_dialogue_ids:
        # Collect predictions for this dialogue across all folds
        fold_predictions = []
        fold_confidences = []
        
        for fold, fold_results in all_fold_results.items():
            if dialogue_id in fold_results['dialogue_ids']:
                idx = fold_results['dialogue_ids'].index(dialogue_id)
                fold_predictions.append(fold_results['predictions'][idx])
                fold_confidences.append(fold_results['confidences'][idx])
        
        if fold_predictions:
            # Majority vote across folds
            vote_counts = Counter(fold_predictions)
            most_common = vote_counts.most_common(1)[0]
            final_prediction = most_common[0]
            
            # Average confidence for the winning class
            winning_confidences = [conf for pred, conf in zip(fold_predictions, fold_confidences) 
                                 if pred == final_prediction]
            final_confidence = np.mean(winning_confidences) if winning_confidences else 0.5
            
            final_results[dialogue_id] = {
                'prediction': final_prediction,
                'prediction_label': class_names[final_prediction],
                'confidence': final_confidence,
                'fold_predictions': fold_predictions,
                'fold_confidences': fold_confidences,
                'vote_counts': dict(vote_counts)
            }
    
    return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='p4g', help='Dataset name')
    parser.add_argument('--input_dir', type=str, default='baselines/data/', help='Input directory')
    parser.add_argument('--local_info', type=str, default=None, help='Local information type')
    parser.add_argument('--global_info', type=str, default=None, help='Global information type')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Model name')
    parser.add_argument('--frac', type=str, default='1', help='Data fraction')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--output_dir', type=str, default='baselines/majority_vote_results/', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--trainable', type=bool, default=True, help='Trainable')

    
    args = parser.parse_args()
    
    # Set up device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Get pairwise mappings and class names
    pairwise_mappings, class_names = get_pairwise_mappings()
    
    print(f"Processing {len(pairwise_mappings)} pairwise classifiers across {args.num_folds} folds")
    print(f"Classes: {class_names}")
    
    # Store results for each fold
    all_fold_results = {}
    
    for fold in range(args.num_folds):
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold}")
        print(f"{'='*50}")
        
        # Load test data for this fold
        test_data_path = f'{args.input_dir}/{args.dataset}/processed/RAT_1_{fold}_test.json'
        
        if not os.path.exists(test_data_path):
            print(f"Warning: Test data not found: {test_data_path}")
            continue
            
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(test_data)} test samples for fold {fold}")
        
        # Create data loader using the existing function
        # We'll use empty train/dev data since we only need test loader
        _, _, test_loader = get_data_loaders(
            test_data, test_data, test_data, tokenizer, args
        )
        
        # Get predictions from all pairwise classifiers
        pairwise_results = []
        
        for pair_info in pairwise_mappings:
            print(f"\nProcessing {pair_info['pair_id']}: {pair_info['class1']} vs {pair_info['class2']}")
            
            # Load the model for this pair
            model = load_pairwise_model(pair_info, fold, args)
            
            if model is None:
                print("MODEL LOADING FAILED!")
            else:
                # Get predictions
                predictions, confidences = predict_pairwise(model, test_loader, device, pair_info)
            
            pairwise_results.append({
                'pair_info': pair_info,
                'predictions': predictions,
                'confidences': confidences
            })
            
            # Clear model from memory
            if model is not None:
                del model
                torch.cuda.empty_cache()
        
        # Perform majority vote for this fold
        final_predictions, final_confidences = majority_vote_classification(
            pairwise_results, pairwise_mappings, num_classes=len(class_names)
        )
        
        # Store results for this fold
        dialogue_ids = [item['dialogue_id'] for item in test_data]
        all_fold_results[fold] = {
            'dialogue_ids': dialogue_ids,
            'predictions': final_predictions,
            'confidences': final_confidences,
            'pairwise_results': pairwise_results
        }
        
        # Save fold-specific results
        fold_output_dir = f"{args.output_dir}/fold_{fold}"
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Save detailed results
        fold_results_df = pd.DataFrame({
            'dialogue_id': dialogue_ids,
            'predicted_class': final_predictions,
            'predicted_label': [class_names[pred] for pred in final_predictions],
            'confidence': final_confidences
        })
        
        fold_results_df.to_csv(f"{fold_output_dir}/majority_vote_results.csv", index=False)
        
        print(f"\nFold {fold} Results:")
        print(f"Total samples: {len(final_predictions)}")
        for i, class_name in enumerate(class_names):
            count = sum(1 for pred in final_predictions if pred == i)
            print(f"  {class_name}: {count} ({count/len(final_predictions)*100:.1f}%)")
    
    # Aggregate results across all folds
    print(f"\n{'='*50}")
    print("Aggregating Results Across All Folds")
    print(f"{'='*50}")
    
    final_aggregated_results = aggregate_across_folds(all_fold_results, class_names)
    
    # Save final aggregated results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create comprehensive results DataFrame
    results_data = []
    for dialogue_id, result in final_aggregated_results.items():
        results_data.append({
            'dialogue_id': dialogue_id,
            'final_prediction': result['prediction'],
            'final_label': result['prediction_label'],
            'final_confidence': result['confidence'],
            'fold_predictions': str(result['fold_predictions']),
            'fold_confidences': str(result['fold_confidences']),
            'vote_counts': str(result['vote_counts'])
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f"{args.output_dir}/final_aggregated_results.csv", index=False)
    
    # Print final statistics
    print(f"\nFinal Aggregated Results:")
    print(f"Total unique dialogues: {len(final_aggregated_results)}")
    
    final_class_counts = Counter([result['prediction'] for result in final_aggregated_results.values()])
    for i, class_name in enumerate(class_names):
        count = final_class_counts.get(i, 0)
        percentage = count / len(final_aggregated_results) * 100 if final_aggregated_results else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Save summary statistics
    with open(f"{args.output_dir}/summary_statistics.txt", 'w') as f:
        f.write("Majority Vote Classification Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Local Info: {args.local_info}\n")
        f.write(f"Global Info: {args.global_info}\n")
        f.write(f"Number of Folds: {args.num_folds}\n")
        f.write(f"Total Pairwise Classifiers: {len(pairwise_mappings)}\n\n")
        
        f.write("Class Distribution:\n")
        for i, class_name in enumerate(class_names):
            count = final_class_counts.get(i, 0)
            percentage = count / len(final_aggregated_results) * 100 if final_aggregated_results else 0
            f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\nTotal Unique Dialogues: {len(final_aggregated_results)}\n")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Files created:")
    print(f"  - final_aggregated_results.csv")
    print(f"  - summary_statistics.txt")
    print(f"  - fold_*/majority_vote_results.csv (for each fold)")

if __name__ == '__main__':
    main()