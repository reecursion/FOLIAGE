import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Path to the directory containing the CSV files
base_dir = "/home/gganeshl/FOLIAGE/src/icl/results/p4g/llama70b"

# Function to calculate metrics for a single file
def analyze_file(filepath):
    print(f"Processing {filepath}...")
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        print(f"Total rows in file: {len(df)}")
        
        # Filter out any rows with missing data
        valid_data = df.dropna(subset=['actual', 'predicted'])

        print(f"Total rows in file after filtering: {len(df)}")
        
        # Convert yes/no to 1/0 for binary classification metrics
        valid_data['actual_binary'] = valid_data['actual'].map({'yes': 1, 'no': 0})
        valid_data['predicted_binary'] = valid_data['predicted'].map({'yes': 1, 'no': 0})
        
        # Calculate metrics
        precision = precision_score(valid_data['actual_binary'], valid_data['predicted_binary'], zero_division=0)
        recall = recall_score(valid_data['actual_binary'], valid_data['predicted_binary'], zero_division=0)
        f1 = f1_score(valid_data['actual_binary'], valid_data['predicted_binary'], zero_division=0)
        
        return {
            'filename': os.path.basename(filepath),
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'count': len(valid_data)
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

# Map file pattern to configuration type based on both directory and filename
def map_file_to_config_type(filepath, filename):
    # Extract the subdirectory information
    if 'baseline' in filepath:
        if 'none_no_intentions' in filename:
            return '(i) Utterance'
        elif 'none_with_intentions' in filename:
            return '(ii) Utterance + Intentions'
    elif 'globalscaffolding' in filepath:
        if 'scd_no_intentions' in filename:
            return '(iii) Utterance + SCD Summary'
        elif 'scm_no_intentions' in filename:
            return '(iv) Utterance + SCM Summary'
    elif 'dualscaffolding' in filepath or 'localscaffolding' in filepath:
        if 'scd_with_intentions' in filename:
            return '(v) Utterance + Intentions + SCD'
        elif 'scm_with_intentions' in filename:
            return '(vi) Utterance + Intentions + SCM'
    
    # Fallback to just filename-based mapping if directory structure doesn't match expected
    if 'none_no_intentions' in filename:
        return '(i) Utterance'
    elif 'none_with_intentions' in filename:
        return '(ii) Utterance + Intentions'
    elif 'scd_no_intentions' in filename:
        return '(iii) Utterance + SCD Summary'
    elif 'scm_no_intentions' in filename:
        return '(iv) Utterance + SCM Summary'
    elif 'scd_with_intentions' in filename:
        return '(v) Utterance + Intentions + SCD'
    elif 'scm_with_intentions' in filename:
        return '(vi) Utterance + Intentions + SCM'
    
    return 'Unknown'

# Extract ratio from filename
def extract_ratio(filename):
    match = re.search(r'p4g_(\d+\.\d+)', filename)
    return match.group(1) if match else None

# Main function to process all files
def process_all_files():
    try:
        # Find all CSV files in the directory
        all_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.csv') and 'final_results' in file:
                    all_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_files)} files to process")
        
        # Process each file
        results = []
        for file_path in all_files:
            try:
                result = analyze_file(file_path)
                result['filepath'] = file_path  # Store the full path for directory identification
                results.append(result)
                print(f"Completed {file_path}: ", result.get('metrics', 'Error'))
            except Exception as e:
                print(f"Error with {file_path}: {e}")
        
        # Organize results for the table
        table_data = {
            'precision': {},
            'recall': {},
            'f1': {}
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
            
            for metric in ['precision', 'recall', 'f1']:
                table_data[metric][config_type][ratio] = result['metrics'][metric]
        
        # Generate combined LaTeX table
        print("\n--- Combined Table for All Metrics ---")
        latex_table = generate_combined_latex_table(table_data, ratios)
        
        # Write LaTeX table to file
        output_file = os.path.join(base_dir, 'persuasion_metrics_table.tex')
        with open(output_file, 'w') as f:
            f.write(latex_table)
        
        print(f"LaTeX table saved to {output_file}")
        
        return latex_table
    except Exception as e:
        print(f"Error processing files: {e}")
        return {'error': str(e)}

# Generate combined LaTeX table for all metrics
def generate_combined_latex_table(data, ratios):
    metric_names = ['Precision', 'Recall', 'F1 Score']
    metric_keys = ['precision', 'recall', 'f1']
    
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
                        cells.append(f"{color_cmd}{{{formatted_value}}}")
            
            row += " & ".join(cells) + " \\\\\n"
            latex_table += row
        
        # Add a midrule between metrics (except after the last one)
        if i < len(metric_names) - 1:
            latex_table += "\\midrule\n"
    
    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += "\\caption{\\textbf{ICL: }Performance metrics across different conversation lengths and configuration types for \\textbf{Llama-3.1-70B} on the \\textbf{Persuasion for Good} dataset}\n"
    latex_table += "\\label{tab:combined_metrics_icl_llama_p4g}\n\\end{table*}"
    
    print(latex_table)
    return latex_table

if __name__ == "__main__":
    process_all_files()