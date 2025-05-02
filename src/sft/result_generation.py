import os
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
import re

# Path to the directory containing the CSV files
base_dir = "/home/gganeshl/FOLIAGE/src/sft/results/craigslistbargain"

# Function to calculate metrics for a single file
def analyze_file(filepath):
    print(f"Processing {filepath}...")
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        print(f"Total rows in file: {len(df)}")
        
        # Extract fold information
        folds = df['fold'].unique()
        
        # Results per fold
        fold_results = {}
        
        for fold in folds:
            fold_data = df[df['fold'] == fold].copy()
            
            # Calculate metrics for this fold
            metrics = calculate_metrics(fold_data)
            fold_results[fold] = metrics
        
        # Average metrics across folds
        avg_metrics = average_metrics_across_folds(fold_results)
        
        return {
            'filename': os.path.basename(filepath),
            'fold_results': fold_results,
            'avg_metrics': avg_metrics
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

# Calculate metrics for a set of predictions
def calculate_metrics(data):
    print(f"  Fold data rows before filtering: {len(data)}")
    
    # Filter out any rows with missing data
    valid_data = data.dropna(subset=['buyer_target', 'seller_target', 'sale_price', 'predicted_final_price'])
    
    print(f"  Fold data rows after filtering: {len(valid_data)}")
    
    # Calculate success metrics
    valid_data['success_sale'] = (valid_data['sale_price'] - valid_data['buyer_target']) / (valid_data['seller_target'] - valid_data['buyer_target'])
    valid_data['success_predicted'] = (valid_data['predicted_final_price'] - valid_data['buyer_target']) / (valid_data['seller_target'] - valid_data['buyer_target'])
    
    # Calculate normalized squared error
    valid_data['normalized_squared_error'] = ((valid_data['sale_price'] - valid_data['predicted_final_price']) ** 2) / valid_data['sale_price']
    
    # Calculate RMSE for success
    success_mse = ((valid_data['success_sale'] - valid_data['success_predicted']) ** 2).mean()
    success_rmse = math.sqrt(success_mse)
    
    # Calculate Pearson correlation for success
    success_pearson, _ = pearsonr(valid_data['success_sale'], valid_data['success_predicted'])
    
    # Calculate NMSE for raw price
    raw_price_nmse = valid_data['normalized_squared_error'].mean()
    
    return {
        'successRMSE': success_rmse,
        'successPearson': success_pearson,
        'rawPriceNMSE': raw_price_nmse,
        'count': len(valid_data)
    }

# Average metrics across folds
def average_metrics_across_folds(fold_results):
    metrics = {
        'successRMSE': 0,
        'successPearson': 0,
        'rawPriceNMSE': 0,
        'totalCount': 0
    }
    
    for fold, fold_metrics in fold_results.items():
        count = fold_metrics['count']
        
        metrics['successRMSE'] += fold_metrics['successRMSE'] * count
        metrics['successPearson'] += fold_metrics['successPearson'] * count
        metrics['rawPriceNMSE'] += fold_metrics['rawPriceNMSE'] * count
        metrics['totalCount'] += count
    
    # Normalize by total count
    if metrics['totalCount'] > 0:
        metrics['successRMSE'] /= metrics['totalCount']
        metrics['successPearson'] /= metrics['totalCount']
        metrics['rawPriceNMSE'] /= metrics['totalCount']
    
    return metrics

# Map file pattern to table row
def map_file_to_config_type(filename):
    if '_none_predictions' in filename:
        return '(i) Utterance'
    elif '_local_predictions' in filename:
        return '(ii) Utterance + Intentions'
    elif '_global_scd_predictions' in filename:
        return '(iii) Utterance + SCD Summary'
    elif '_global_scm_predictions' in filename:
        return '(iv) Utterance + SCM Summary'
    elif '_both_scd_predictions' in filename:
        return '(v) Utt + Intentions + SCD Summary'
    elif '_both_scm_predictions' in filename:
        return '(vi) Utt + Intentions + SCM Summary'
    else:
        return 'Unknown'

# Extract ratio from filename
def extract_ratio(filename):
    match = re.search(r'cb_ratio_(\d+\.\d+)', filename)
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
        
        # Process each file
        results = []
        for file_path in all_files:
            try:
                result = analyze_file(file_path)
                results.append(result)
                print(f"Completed {file_path}: ", result.get('avg_metrics', 'Error'))
            except Exception as e:
                print(f"Error with {file_path}: {e}")
        
        # Organize results for the table
        table_data = {
            'successRMSE': {},
            'successPearson': {},
            'rawPriceNMSE': {}
        }
        
        # Initialize structure
        config_types = [
            '(i) Utterance',
            '(ii) Utterance + Intentions',
            '(iii) Utterance + SCD Summary',
            '(iv) Utterance + SCM Summary',
            '(v) Utt + Intentions + SCD Summary',
            '(vi) Utt + Intentions + SCM Summary'
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
            
            config_type = map_file_to_config_type(result['filename'])
            ratio = extract_ratio(result['filename'])
            
            if not config_type or not ratio or ratio not in ratios:
                continue
            
            table_data['successRMSE'][config_type][ratio] = result['avg_metrics']['successRMSE']
            table_data['successPearson'][config_type][ratio] = result['avg_metrics']['successPearson']
            table_data['rawPriceNMSE'][config_type][ratio] = result['avg_metrics']['rawPriceNMSE']
        
        # Generate combined LaTeX table
        print("\n--- Combined Table for All Metrics ---")
        generate_combined_latex_table(table_data, ratios)
        
        return results
    except Exception as e:
        print(f"Error processing files: {e}")
        return {'error': str(e)}

# Generate combined LaTeX table for all metrics
def generate_combined_latex_table(data, ratios):
    metric_names = ['Success RMSE', 'Success Pearson', 'Raw Price NMSE']
    metric_keys = ['successRMSE', 'successPearson', 'rawPriceNMSE']
    
    # Start the table
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{ll" + "c" * len(ratios) + "}\n\\hline\n"
    latex_table += "\\textbf{Metric} & \\textbf{Config} & " + " & ".join([f"\\textbf{{{float(r) * 100:.1f}\\%}}" for r in ratios]) + " \\\\\n\\hline\n"
    
    # For each metric
    for i, metric_name in enumerate(metric_names):
        metric_key = metric_keys[i]
        metric_data = data[metric_key]
        
        # Get configs that have data for this metric
        config_types = [ct for ct in metric_data.keys() if any(metric_data[ct].get(r) is not None for r in ratios)]
        
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
                        # For RMSE and NMSE, lower is better (green)
                        # For Pearson, higher is better (green)
                        is_improvement = metric_key == 'successPearson'
                        if is_improvement:
                            is_better = value > baseline_values[ratio]
                        else:
                            is_better = value < baseline_values[ratio]
                        
                        color_cmd = '\\cellcolor{green!25}' if is_better else '\\cellcolor{red!25}'
                        cells.append(f"{color_cmd}{{{formatted_value}}}")
            
            row += " & ".join(cells) + " \\\\\n"
            latex_table += row
        
        # Add a midrule between metrics (except after the last one)
        if i < len(metric_names) - 1:
            latex_table += "\\midrule\n"
    
    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += "\\caption{Performance metrics across different conversation lengths and configuration types.}\n"
    latex_table += "\\label{tab:combined_metrics}\n\\end{table}"
    
    print(latex_table)
    return latex_table

# Legacy function for individual tables - kept for reference
def generate_latex_table(data, metric_name, ratios):
    config_types = [ct for ct in data.keys() if any(data[ct].get(r) is not None for r in ratios)]
    
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * len(ratios) + "}\n\\hline\n"
    latex_table += "\\textbf{Config} & " + " & ".join([f"\\textbf{{{float(r) * 100:.1f}\\%}}" for r in ratios]) + " \\\\\n\\hline\n"
    
    # Get baseline values for color coding
    baseline_type = '(i) Utterance'
    baseline_values = {}
    
    for ratio in ratios:
        if baseline_type in data and ratio in data[baseline_type] and data[baseline_type][ratio] is not None:
            baseline_values[ratio] = data[baseline_type][ratio]
    
    # Add rows
    for config_type in config_types:
        latex_table += f"{config_type} & "
        
        cells = []
        for ratio in ratios:
            value = data[config_type].get(ratio)
            
            if value is None:
                cells.append('-')
            else:
                # Format value (round to 4 decimal places)
                formatted_value = f"{value:.4f}"
                
                # Color compared to baseline
                if config_type == baseline_type or ratio not in baseline_values:
                    cells.append(formatted_value)
                else:
                    # For RMSE and NMSE, lower is better (green)
                    # For Pearson, higher is better (green)
                    is_improvement = 'Pearson' in metric_name
                    if is_improvement:
                        is_better = value > baseline_values[ratio]
                    else:
                        is_better = value < baseline_values[ratio]
                    
                    color_cmd = '\\textcolor{green}' if is_better else '\\textcolor{red}'
                    cells.append(f"{color_cmd}{{{formatted_value}}}")
        
        latex_table += " & ".join(cells) + " \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += f"\\caption{{{metric_name} across different conversation lengths and configuration types.}}\n"
    latex_table += f"\\label{{tab:{metric_name.lower().replace(' ', '_')}}}\n\\end{table}"
    
    return latex_table

if __name__ == "__main__":
    process_all_files()