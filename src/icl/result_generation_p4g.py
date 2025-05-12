import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Path to the directory containing the CSV files
base_dir = "/users/rithviksenthil/desktop/FOLIAGE/src/icl/results/p4g/llama70b/"

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
        precision = precision_score(valid_data['actual_binary'], valid_data['predicted_binary'], zero_division=0, average="macro")
        recall = recall_score(valid_data['actual_binary'], valid_data['predicted_binary'], zero_division=0, average="macro")
        f1 = f1_score(valid_data['actual_binary'], valid_data['predicted_binary'], zero_division=0, average="macro")
        
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
    # Extract configuration based on filename and filepath patterns
    config_type = 'Unknown'
    
    # Extract summary type
    if 'none' in filename:
        summary_prefix = '(i)'
    elif 'scd' in filename:
        summary_prefix = '(ii)'
    elif 'scm' in filename:
        summary_prefix = '(iii)'
    elif 'traditional' in filename:
        summary_prefix = '(iv)'
    else:
        summary_prefix = '(?)'
    
    # Extract intentions information
    if 'with_intentions' in filename or '_true' in filename:
        intentions_suffix = 'with Intentions'
    else:
        intentions_suffix = 'no Intentions'
    
    # Combine for full config label
    if summary_prefix == '(i)':
        if 'no Intentions' in intentions_suffix:
            config_type = '(i) Utterance'
        else:
            config_type = '(ii) Utterance + Intentions'
    elif summary_prefix == '(ii)':
        if 'no Intentions' in intentions_suffix:
            config_type = '(iii) Utterance + SCD'
        else:
            config_type = '(vi) Utterance + Intentions + SCD'
    elif summary_prefix == '(iii)':
        if 'no Intentions' in intentions_suffix:
            config_type = '(iv) Utterance + SCM'
        else:
            config_type = '(vii) Utterance + Intentions + SCM'
    elif summary_prefix == '(iv)':
        if 'no Intentions' in intentions_suffix:
            config_type = '(v) Utterance + Traditional'
        else:
            config_type = '(viii) Utterance + Intentions + Traditional'

    return config_type

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
            '(iii) Utterance + SCD',
            '(iv) Utterance + SCM',
            '(v) Utterance + Traditional',
            '(vi) Utterance + Intentions + SCD',
            '(vii) Utterance + Intentions + SCM',
            '(viii) Utterance + Intentions + Traditional'
        ]
        
        ratios = ['0.25', '0.375', '0.5', '0.625', '0.75']
        
        # Initialize empty table
        for metric_type in table_data.keys():
            table_data[metric_type] = {}
            
            for config_type in config_types:
                table_data[metric_type][config_type] = {}
                
                for ratio in ratios:
                    table_data[metric_type][config_type][ratio] = 0
        
        # Fill in table data
        for result in results:
            if 'error' in result:
                continue
            
            config_type = map_file_to_config_type(result['filepath'], result['filename'])
            ratio = extract_ratio(result['filename'])
            
            if not config_type or not ratio or ratio not in ratios:
                continue
            
            for metric in ['precision', 'recall', 'f1']:
                table_data[metric][config_type][ratio] += result['metrics'][metric]

        for metric in table_data:
            for config_type in table_data[metric]:
                for ratio in table_data[metric][config_type]:
                        table_data[metric][config_type][ratio] /= 3
        
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
