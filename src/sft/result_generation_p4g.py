import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
        fold_metrics = {}
        for fold, fold_data in valid_data.groupby('fold'):
            # Skip if fold has no data
            if len(fold_data) == 0:
                continue
            
            print(f"\nAnalyzing Fold {fold}: {len(fold_data)} samples")
                
            actual = fold_data['actual_binary'].values
            predicted = fold_data['predicted_binary'].values
            
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
            
            print(f"  Actually donated: {true_count}/{len(fold_data)} ({true_count/len(fold_data)*100:.1f}%)")
            print(f"  Predicted donations: {yes_count}/{len(fold_data)} ({yes_count/len(fold_data)*100:.1f}%)")
            
            fold_metrics[fold] = {
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
                'total': len(fold_data)
            }
        
        # Calculate average metrics across all folds
        if fold_metrics:
            avg_metrics = {
                'precision': np.mean([m['precision'] for m in fold_metrics.values()]),
                'recall': np.mean([m['recall'] for m in fold_metrics.values()]),
                'f1': np.mean([m['f1'] for m in fold_metrics.values()]),
                'accuracy': np.mean([m['accuracy'] for m in fold_metrics.values()]),
                'total': sum([m['total'] for m in fold_metrics.values()]),
                'donation_true': sum([m['donation_true'] for m in fold_metrics.values()]),
                'donation_false': sum([m['donation_false'] for m in fold_metrics.values()]),
                'predicted_yes': sum([m['predicted_yes'] for m in fold_metrics.values()]),
                'predicted_no': sum([m['predicted_no'] for m in fold_metrics.values()])
            }
            fold_metrics['average'] = avg_metrics
            
        return {
            'filename': os.path.basename(filepath),
            'fold_metrics': fold_metrics,
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
    
    if 'none' in filename:
        return '(i) Utterance'
    elif 'local' in filename:
        return '(ii) Utterance + Intentions'
    elif 'global_scd' in filename:
        return '(iii) Utterance + SCD Summary'
    elif 'global_scm' in filename:
        return '(iv) Utterance + SCM Summary'
    elif 'both_scd' in filename:
        return '(v) Utterance + Intentions + SCD'
    elif 'both_scm' in filename:
        return '(vi) Utterance + Intentions + SCM'
    
    return 'Unknown'

# Extract ratio from filename
def extract_ratio(filename):
    match = re.search(r'ratio_(\d+\.\d+)', filename)
    return match.group(1) if match else None

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
        
        # Process each file
        results = []
        for file_path in all_files:
            try:
                result = analyze_file(file_path)
                result['filepath'] = file_path  # Store the full path for directory identification
                results.append(result)
                print(f"Completed {file_path}: ", result.get('fold_metrics', 'Error'))
            except Exception as e:
                print(f"Error with {file_path}: {e}")
        
        # Organize results for the table
        table_data = {
            'precision': {},
            'recall': {},
            'f1': {},
            'accuracy': {}
        }
        
        # Initialize structure
        config_types = [
            '(i) Utterance',
            '(ii) Utterance + Intentions',
            '(iii) Utterance + SCD Summary',
            '(iv) Utterance + SCM Summary',
            '(v) Utterance + Intentions + SCD',
            '(vi) Utterance + Intentions + SCM'
        ]
        
        ratios = ['0.25', '0.375', '0.5', '0.625', '0.75']
        
        # Initialize empty table
        for metric_type in table_data.keys():
            table_data[metric_type] = {}
            
            for config_type in config_types:
                table_data[metric_type][config_type] = {}
                
                for ratio in ratios:
                    table_data[metric_type][config_type][ratio] = None
        
        # Fill in table data
        for result in results:
            if 'error' in result:
                continue
            
            config_type = map_file_to_config_type(result['filepath'], result['filename'])
            ratio = extract_ratio(result['filename'])
            
            if not config_type or not ratio or ratio not in ratios:
                continue
            
            # Get average metrics across all folds
            fold_metrics = result.get('fold_metrics', {})
            if 'average' in fold_metrics:
                avg_metrics = fold_metrics['average']
                
                for metric in ['precision', 'recall', 'f1', 'accuracy']:
                    if metric in avg_metrics:
                        table_data[metric][config_type][ratio] = avg_metrics[metric]
        
        # Generate combined LaTeX table
        print("\n--- Combined Table for All Metrics ---")
        latex_table = generate_combined_latex_table(table_data, ratios)
        
        # Initialize fold report (added to avoid undefined variable)
        fold_report = "Fold sizes summary not implemented"
        
        return {
            'combined_table': latex_table,
            'fold_report': fold_report
        }
    except Exception as e:
        print(f"Error processing files: {e}")
        return {'error': str(e)}
        
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
        # Otherwise process all files
        result = process_all_files()
