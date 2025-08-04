import pandas as pd
import json
import argparse
import numpy as np
from typing import Dict, Any

def determine_ground_truth_class(dialogue_data: Dict[str, Any]) -> int:
    """
    Determine the ground truth class based on stated intention and actual donation
    
    Classes:
        0: 'expressed_donate_did'      - Expressed intention of donating but did not donate
        1: 'expressed_donate_donated'  - Expressed intention and donated  
        2: 'no_express_donated'        - Did not express intention but donated
        3: 'no_express_no_donate'      - Did not express intention and did not donate
        4: 'unclear_donated'           - Unclear but donated
        5: 'unclear_no_donate'         - Unclear but did not donate
    """
    
    stated_intention = dialogue_data.get('stated_intention', '').lower()
    actual_donation = dialogue_data.get('actual_donation', False)
    
    # Handle the mapping based on stated intention and actual donation
    if stated_intention == 'donate':
        if actual_donation:
            return 1  # expressed_donate_donated
        else:
            return 0  # expressed_donate_did
    elif stated_intention == 'not_donate' or stated_intention == 'no_donate':
        if actual_donation:
            return 2  # no_express_donated (refused but actually donated)
        else:
            return 3  # no_express_no_donate
    elif stated_intention == 'unclear' or stated_intention == '' or stated_intention is None:
        if actual_donation:
            return 4  # unclear_donated
        else:
            return 5  # unclear_no_donate
    else:
        # Handle any other cases as unclear
        if actual_donation:
            return 4  # unclear_donated
        else:
            return 5  # unclear_no_donate

def get_class_names():
    """Return the class names mapping"""
    return [
        'expressed_donate_did',      # 0: Expressed intention of donating but did not donate
        'expressed_donate_donated',  # 1: Expressed intention and donated  
        'no_express_donated',        # 2: Did not express intention but donated
        'no_express_no_donate',      # 3: Did not express intention and did not donate
        'unclear_donated',           # 4: Unclear but donated
        'unclear_no_donate'          # 5: Unclear but did not donate
    ]

def add_ground_truth_labels(predictions_csv_path: str, analysis_json_path: str, output_csv_path: str):
    """
    Add ground truth labels to the predictions CSV based on the analysis JSON
    
    Args:
        predictions_csv_path: Path to the predictions CSV file
        analysis_json_path: Path to the analysis JSON file
        output_csv_path: Path to save the updated CSV with ground truth labels
    """
    
    # Load the predictions CSV
    predictions_df = pd.read_csv(predictions_csv_path)
    print(f"Loaded predictions CSV with {len(predictions_df)} rows")
    
    # Load the analysis JSON
    with open(analysis_json_path, 'r') as f:
        analysis_data = json.load(f)
    
    detailed_results = analysis_data.get('detailed_results', [])
    print(f"Loaded analysis JSON with {len(detailed_results)} detailed results")
    
    # Create a mapping from dialogue_id to ground truth data
    dialogue_to_gt = {}
    for result in detailed_results:
        dialogue_id = result.get('dialogue_id')
        if dialogue_id:
            dialogue_to_gt[dialogue_id] = result
    
    print(f"Created ground truth mapping for {len(dialogue_to_gt)} dialogues")
    
    # Get class names
    class_names = get_class_names()
    
    # Add ground truth columns
    ground_truth_labels = []
    ground_truth_names = []
    has_ground_truth = []
    stated_intentions = []
    actual_donations = []
    inconsistencies = []
    
    missing_dialogues = []
    
    for idx, row in predictions_df.iterrows():
        dialogue_id = row['dialogue_id']
        
        if dialogue_id in dialogue_to_gt:
            gt_data = dialogue_to_gt[dialogue_id]
            
            # Determine ground truth class
            gt_class = determine_ground_truth_class(gt_data)
            ground_truth_labels.append(gt_class)
            ground_truth_names.append(class_names[gt_class])
            has_ground_truth.append(True)
            
            # Add additional information
            stated_intentions.append(gt_data.get('stated_intention', ''))
            actual_donations.append(gt_data.get('actual_donation', False))
            inconsistencies.append(gt_data.get('inconsistency', False))
            
        else:
            # No ground truth data available for this dialogue
            ground_truth_labels.append(-1)  # Use -1 to indicate missing
            ground_truth_names.append('missing')
            has_ground_truth.append(False)
            stated_intentions.append('')
            actual_donations.append(False)
            inconsistencies.append(False)
            missing_dialogues.append(dialogue_id)
    
    # Add the new columns to the DataFrame
    predictions_df['ground_truth_label'] = ground_truth_labels
    predictions_df['ground_truth_name'] = ground_truth_names
    predictions_df['has_ground_truth'] = has_ground_truth
    predictions_df['stated_intention'] = stated_intentions
    predictions_df['actual_donation'] = actual_donations
    predictions_df['inconsistency'] = inconsistencies
    
    # Add accuracy column (only for cases where we have ground truth)
    predictions_df['correct_prediction'] = (
        (predictions_df['final_prediction'] == predictions_df['ground_truth_label']) & 
        predictions_df['has_ground_truth']
    )
    
    # Save the updated CSV
    predictions_df.to_csv(output_csv_path, index=False)
    print(f"Saved updated predictions to: {output_csv_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    total_predictions = len(predictions_df)
    has_gt_count = sum(has_ground_truth)
    missing_gt_count = total_predictions - has_gt_count
    
    print(f"Total predictions: {total_predictions}")
    print(f"With ground truth: {has_gt_count}")
    print(f"Missing ground truth: {missing_gt_count}")
    
    if missing_gt_count > 0:
        print(f"\nMissing ground truth for dialogues:")
        for dialogue_id in missing_dialogues[:10]:  # Show first 10
            print(f"  - {dialogue_id}")
        if len(missing_dialogues) > 10:
            print(f"  ... and {len(missing_dialogues) - 10} more")
    
    # Calculate accuracy for cases with ground truth
    if has_gt_count > 0:
        correct_predictions = sum(predictions_df['correct_prediction'])
        accuracy = correct_predictions / has_gt_count
        print(f"\nOverall Accuracy: {accuracy:.4f} ({correct_predictions}/{has_gt_count})")
        
        # Print confusion matrix information
        print(f"\nGround Truth Distribution:")
        gt_counts = predictions_df[predictions_df['has_ground_truth']]['ground_truth_name'].value_counts()
        for class_name, count in gt_counts.items():
            percentage = count / has_gt_count * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nPrediction Distribution (for cases with GT):")
        pred_counts = predictions_df[predictions_df['has_ground_truth']]['final_label'].value_counts()
        for class_name, count in pred_counts.items():
            percentage = count / has_gt_count * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Class-wise accuracy
        print(f"\nClass-wise Accuracy:")
        for class_name in class_names:
            class_mask = (predictions_df['ground_truth_name'] == class_name) & predictions_df['has_ground_truth']
            if class_mask.sum() > 0:
                class_correct = (
                    (predictions_df['final_label'] == class_name) & class_mask
                ).sum()
                class_total = class_mask.sum()
                class_accuracy = class_correct / class_total
                print(f"  {class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")
        
        # Inconsistency analysis
        inconsistent_cases = predictions_df[predictions_df['inconsistency'] == True]
        if len(inconsistent_cases) > 0:
            print(f"\nInconsistency Analysis:")
            print(f"Total inconsistent cases: {len(inconsistent_cases)}")
            inconsistent_accuracy = inconsistent_cases['correct_prediction'].sum() / len(inconsistent_cases)
            print(f"Accuracy on inconsistent cases: {inconsistent_accuracy:.4f}")
    
    return predictions_df

def main():
    parser = argparse.ArgumentParser(description='Add ground truth labels to predictions CSV')
    parser.add_argument('--predictions_csv', default="baselines/majority_vote_results/final_aggregated_results.csv", type=str,
                       help='Path to the predictions CSV file')
    parser.add_argument('--analysis_json', default="baselines/majority_vote_results/persuasion_analysis_results.json", type=str,
                       help='Path to the analysis JSON file')
    parser.add_argument('--output_csv', default="baselines/output.csv", type=str,
                       help='Path to save the updated CSV file')
    
    args = parser.parse_args()
    
    # Add ground truth labels
    updated_df = add_ground_truth_labels(
        args.predictions_csv, 
        args.analysis_json, 
        args.output_csv
    )
    
    print(f"\nUpdated CSV columns:")
    for col in updated_df.columns:
        print(f"  - {col}")

if __name__ == '__main__':
    main()