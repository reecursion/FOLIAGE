import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from collections import defaultdict

# Path to the directory containing the CSV files
base_dir = "/home/gganeshl/FOLIAGE/src/sft/results/p4g"

# Function to calculate metrics for a single file
def analyze_file(filepath):
    print(f"Processing {filepath}...")
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        print(f"Total rows in file: {len(df)}")
        
        # Extract fold information before filtering
        folds_before = df['fold'].value_counts().to_dict()
        print("\nFold distribution before filtering:")
        for fold, count in sorted(folds_before.items()):
            print(f"  Fold {fold}: {count} samples")
        
        # Filter out any rows with missing data
        valid_data = df.dropna(subset=['label', 'predicted_label'])
        print(f"\nTotal rows after filtering: {len(valid_data)}")
        
        # Extract fold information after filtering
        folds_after = valid_data['fold'].value_counts().to_dict()
        print("\nFold distribution after filtering:")
        for fold, count in sorted(folds_after.items()):
            print(f"  Fold {fold}: {count} samples")
            
        # Store fold counts for reporting
        fold_counts_before = folds_before
        fold_counts_after = folds_after
        
        # Convert True/False and YES/NO to 1/0 for binary classification metrics
        valid_data['actual_binary'] = valid_data['label']
        valid_data['predicted_binary'] = valid_data['predicted_label']
        
        # Group by fold and calculate metrics for each fold
        # fold_metrics = {}
        # for fold, fold_data in valid_data.groupby('fold'):
            # Skip if fold has no data
            # if len(fold_data) == 0:
            #     continue
            
            # print(f"\nAnalyzing Fold {fold}: {len(fold_data)} samples")
                
        actual = valid_data['actual_binary'].values
        predicted = valid_data['predicted_binary'].values
            
        # Calculate metrics
        precision = precision_score(actual, predicted, zero_division=0, average="macro")
        recall = recall_score(actual, predicted, zero_division=0, average="macro")
        f1 = f1_score(actual, predicted, zero_division=0, average="macro")
        accuracy = accuracy_score(actual, predicted)
        
        # Count true/false positives/negatives
        tp = ((actual == 1) & (predicted == 1)).sum()
        fp = ((actual == 0) & (predicted == 1)).sum()
        tn = ((actual == 0) & (predicted == 0)).sum()
        fn = ((actual == 1) & (predicted == 0)).sum()
        
        # Count actual donations and predictions
        true_count = (actual == 1).sum()
        false_count = (actual == 0).sum()
        yes_count = (predicted == 1).sum()
        no_count = (predicted == 0).sum()
            
        # print(f"  Actually donated: {true_count}/{len(fold_data)} ({true_count/len(fold_data)*100:.1f}%)")
        # print(f"  Predicted donations: {yes_count}/{len(fold_data)} ({yes_count/len(fold_data)*100:.1f}%)")
            
        final_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'donation_true': true_count,
            'donation_false': false_count,
            'predicted_yes': yes_count,
            'predicted_no': no_count,
            'total': len(valid_data)
        }
        
        # Calculate average metrics across all folds
        metrics = {}
        metrics['average'] = final_metrics

            
        return {
            'filename': os.path.basename(filepath),
            'fold_metrics': metrics,
            'fold_counts_before': fold_counts_before,
            'fold_counts_after': fold_counts_after
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

# Map file pattern to configuration type based on both directory and filename
def map_file_to_config_type(filepath, filename):
    if '_none_predictions' in filename:
        return '(i) Utterance'
    elif '_local_predictions' in filename:
        return '(ii) Utterance + Intentions'
    elif '_global_scd_predictions' in filename:
        return '(iii) Utterance + SCD Summary'
    elif '_global_scm_predictions' in filename:
        return '(iv) Utterance + SCM Summary'
    elif '_global_traditional_predictions' in filename:
        return '(v) Utterance + Traditional Summary'
    elif '_both_scd_predictions' in filename:
        return '(vi) Utt + Intentions + SCD Summary'
    elif '_both_scm_predictions' in filename:
        return '(vii) Utt + Intentions + SCM Summary'
    elif '_both_traditional_predictions' in filename:
        return '(viii) Utt + Intentions + Traditional Summary'
    else:
        return 'Unknown'

# Extract ratio from filename
def extract_ratio(filename):
    match = re.search(r'ratio_(\d+\.\d+)', filename)
    return match.group(1) if match else None

def create_individual_confusion_matrix(y_true, y_pred, config_type, ratio, output_dir):
    """Create individual confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Donation', 'Donation'],
                yticklabels=['No Donation', 'Donation'])
    
    plt.title(f'{config_type}\nRatio: {float(ratio)*100:.1f}% (All Seeds Combined)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save individual plot
    safe_name = config_type.replace('(', '').replace(')', '').replace(' ', '_').replace('+', 'plus')
    filename = f"{safe_name}_ratio_{ratio}_confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")
    return cm

def create_combined_confusion_matrices(output_dir):
    """Create combined plot with all confusion matrices."""
    # Get all saved confusion matrix data by re-reading files
    grouped_files = defaultdict(list)
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'predictions' in file:
                file_path = os.path.join(root, file)
                filename = os.path.basename(file_path)
                config_type = map_file_to_config_type(file_path, filename)
                ratio = extract_ratio(filename)
                
                if config_type != 'Unknown' and ratio:
                    key = (config_type, ratio)
                    grouped_files[key].append(file_path)
    
    # Get unique configs and ratios
    configs = sorted(set([key[0] for key in grouped_files.keys()]))
    ratios = sorted(set([key[1] for key in grouped_files.keys()]))
    
    # Create subplot grid
    fig, axes = plt.subplots(len(configs), len(ratios), figsize=(4*len(ratios), 3*len(configs)))
    
    if len(configs) == 1:
        axes = axes.reshape(1, -1)
    if len(ratios) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, config in enumerate(configs):
        for j, ratio in enumerate(ratios):
            if (config, ratio) in grouped_files:
                # Combine data for this config/ratio
                all_actual = []
                all_predicted = []
                
                for file_path in grouped_files[(config, ratio)]:
                    df = pd.read_csv(file_path)
                    valid_data = df.dropna(subset=['label', 'predicted_label'])
                    all_actual.extend(valid_data['label'].tolist())
                    all_predicted.extend(valid_data['predicted_label'].tolist())
                
                cm = confusion_matrix(all_actual, all_predicted)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                          xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                          ax=axes[i, j], cbar=False)
                
                short_config = config.split(') ')[1] if ') ' in config else config
                axes[i, j].set_title(f'{short_config}\n{float(ratio)*100:.1f}%', fontsize=10)
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined plot: combined_confusion_matrices.png")

# Main function to process all files
def process_all_files():
    try:
        # Find all CSV files in the directory
        all_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.csv') and 'predictions' in file:
                    all_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_files)} files to process")
        
        # Print the list of files being processed
        for i, file_path in enumerate(all_files):
            print(f"{i+1}. {os.path.basename(file_path)}")
        
        # Group files by configuration and ratio (aggregate across seeds)
        grouped_files = defaultdict(list)
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            config_type = map_file_to_config_type(file_path, filename)
            ratio = extract_ratio(filename)
            
            if config_type != 'Unknown' and ratio:
                key = (config_type, ratio)
                grouped_files[key].append(file_path)
        
        print(f"\nFound {len(grouped_files)} unique config/ratio combinations")
        
        # Create output directory for confusion matrices
        output_dir = os.path.join(base_dir, "confusion_matrices")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Process each group - combine all seeds
        all_confusion_matrices = {}
        
        for (config_type, ratio), file_paths in grouped_files.items():
            print(f"\nProcessing {config_type} - Ratio {ratio}")
            print(f"  Files: {len(file_paths)} (from different seeds)")
            
            # Combine predictions from all seeds
            all_actual = []
            all_predicted = []
            
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                valid_data = df.dropna(subset=['label', 'predicted_label'])
                all_actual.extend(valid_data['label'].tolist())
                all_predicted.extend(valid_data['predicted_label'].tolist())
            
            print(f"  Combined predictions: {len(all_actual)}")
            
            # Create individual confusion matrix
            cm = create_individual_confusion_matrix(all_actual, all_predicted, config_type, ratio, output_dir)
            all_confusion_matrices[(config_type, ratio)] = cm
        
        # Create combined visualization
        print(f"\nCreating combined confusion matrix visualization...")
        create_combined_confusion_matrices(output_dir)
        
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total combinations processed: {len(all_confusion_matrices)}")
        print(f"Individual confusion matrices saved: {len(all_confusion_matrices)}")
        print(f"Combined visualization saved: 1")
        print(f"All files saved to: {output_dir}")
        
        return all_confusion_matrices
        
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Generate combined LaTeX table for all metrics
def generate_combined_latex_table(data, ratios):
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    metric_keys = ['precision', 'recall', 'f1', 'accuracy']
    
    # Start the table
    latex_table = "\\begin{table*}[ht]\n\\centering\n\\begin{tabular}{ll" + "c" * len(ratios) + "}\n\\hline\n"
    latex_table += "\\textbf{Metric} & \\textbf{Config} & " + " & ".join([f"\\textbf{{{float(r) * 100:.1f}\\%}}" for r in ratios]) + " \\\\\n\\hline\n"
    
    # For each metric
    for i, metric_name in enumerate(metric_names):
        metric_key = metric_keys[i]
        metric_data = data[metric_key]
        
        # Get configs that have data for this metric
        config_types = [ct for ct in metric_data.keys() if any(metric_data[ct].get(r) is not None for r in ratios)]
        # Sort by configuration number
        config_types.sort(key=lambda x: x.split('(')[1].split(')')[0] if '(' in x else '')
        
        # Get baseline values for color coding
        baseline_type = '(i) Utterance'
        baseline_values = {}
        
        for ratio in ratios:
            if baseline_type in metric_data and ratio in metric_data[baseline_type] and metric_data[baseline_type][ratio] is not None:
                baseline_values[ratio] = metric_data[baseline_type][ratio]
        
        # Add metric name to first row only
        first_config = True
        
        # Add rows for each config
        for config_type in config_types:
            row = ""
            
            # Add metric name only to first row of the metric section
            if first_config:
                row += f"\\multirow{{{len(config_types)}}}{{*}}{{\\textbf{{{metric_name}}}}} & {config_type} & "
                first_config = False
            else:
                row += f" & {config_type} & "
            
            cells = []
            for ratio in ratios:
                value = metric_data[config_type].get(ratio)
                
                if value is None:
                    cells.append('-')
                else:
                    # Format value (round to 4 decimal places)
                    formatted_value = f"{value:.4f}"
                    
                    # Color compared to baseline
                    if config_type == baseline_type or ratio not in baseline_values:
                        cells.append(formatted_value)
                    else:
                        # For all these metrics, higher is better
                        if value == baseline_values[ratio]:
                            color_cmd = '\\cellcolor{yellow!25}'
                        else:
                            is_better = value > baseline_values[ratio]
                            color_cmd = '\\cellcolor{green!25}' if is_better else '\\cellcolor{red!25}'
                        cells.append(f"{color_cmd}{formatted_value}")
            
            row += " & ".join(cells) + " \\\\\n"
            latex_table += row
        
        # Add a midrule between metrics (except after the last one)
        if i < len(metric_names) - 1:
            latex_table += "\\midrule\n"
    
    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += "\\caption{\\textbf{SFT: }Performance metrics across different conversation lengths and configuration types for \\textbf{Llama-3.1-70B} on the \\textbf{Persuasion for Good} dataset}\n"
    latex_table += "\\label{tab:combined_metrics_sft_llama_p4g}\n\\end{table*}"
    
    print(latex_table)
    return latex_table

# For processing a single file directly
def process_single_file(file_path):
    try:
        result = analyze_file(file_path)
        
        # Print fold-wise metrics
        print("\n--- Fold-wise Metrics ---")
        fold_metrics = result.get('fold_metrics', {})
        for fold, metrics in fold_metrics.items():
            if fold == 'average':
                print(f"\nAverage across all folds:")
            else:
                print(f"\nFold {fold}:")
                
            print(f"  Total samples: {metrics.get('total', 0)}")
            print(f"  Actually donated: {metrics.get('donation_true', 0)}/{metrics.get('total', 0)} ({metrics.get('donation_true', 0)/metrics.get('total', 1)*100:.1f}%)")
            print(f"  Predicted donations: {metrics.get('predicted_yes', 0)}/{metrics.get('total', 0)} ({metrics.get('predicted_yes', 0)/metrics.get('total', 1)*100:.1f}%)")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # If a specific file is provided, process just that file
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = process_single_file(file_path)
    else:
        # Process all files and create confusion matrices
        print("Processing all files and creating confusion matrices...")
        confusion_matrices = process_all_files()