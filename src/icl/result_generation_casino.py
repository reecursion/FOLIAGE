import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error

# Path to the directory containing the CSV files
base_dir = "/home/gganeshl/FOLIAGE/src/icl/results/casino"  # Path to result files

# Path to the preferences CSV file - now using ratio pattern in filename
preferences_csv_path = "/home/gganeshl/FOLIAGE/datasets/casino/final/ratio_0.5.csv"

# Command line argument parsing
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process casino dataset metrics.')
    parser.add_argument('--results_dir', type=str, default="/home/gganeshl/FOLIAGE/src/icl/results/casino",
                        help='Directory containing result CSV files')
    parser.add_argument('--preferences_file', type=str, 
                        default="/home/gganeshl/FOLIAGE/datasets/casino/final/ratio_0.5.csv",
                        help='CSV file with agent preferences')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path for LaTeX table (default: {results_dir}/casino_metrics_table.tex)')
    return parser.parse_args()

# Function to calculate utilities based on preferences
def calculate_utility(food, water, firewood, preferences):
    """
    Calculate utility score based on resource allocation and preferences
    preferences is a dict with keys 'food', 'water', 'firewood' and values 5 (high), 4 (med), 3 (low)
    """
    return (preferences['food'] * food + 
            preferences['water'] * water + 
            preferences['firewood'] * firewood)

# Extract ratio from filename - new function similar to paste2.txt
def extract_ratio(filename):
    match = re.search(r'casino_(\d+\.\d+)', filename)
    if match:
        return match.group(1)
    
    # Alternative pattern if needed
    match = re.search(r'ratio_(\d+\.\d+)', filename)
    return match.group(1) if match else "0.5"  # Default to 0.5 if not found

# Load the preferences from the CSV file based on specified ratio
def load_preferences(ratio="0.5"):
    # Construct the preferences file path with the correct ratio
    ratio_preferences_path = preferences_csv_path.replace("ratio_0.5", f"ratio_{ratio}")
    print(f"Loading preferences data from {ratio_preferences_path}")
    
    try:
        prefs_df = pd.read_csv(ratio_preferences_path)
        
        # Create a dictionary to store preferences for each dialogue
        dialogue_preferences = {}
        
        # Track duplicate dialogue IDs
        duplicate_count = 0
        
        # Process each dialogue in the dataframe
        for idx, row in prefs_df.iterrows():
            dialogue_id = row['dialogue_id']
            
            # Check if this dialogue_id has already been processed
            if dialogue_id in dialogue_preferences:
                duplicate_count += 1
                # print(f"Warning: Duplicate dialogue_id {dialogue_id} found (row {idx}). Using first occurrence only.")
                continue
            
            # Extract preferences for each agent
            agent1_prefs = {
                'high_item': row.get('mturk_agent_1_high_item', '').lower() if pd.notna(row.get('mturk_agent_1_high_item', '')) else '',
                'medium_item': row.get('mturk_agent_1_medium_item', '').lower() if pd.notna(row.get('mturk_agent_1_medium_item', '')) else '',
                'low_item': row.get('mturk_agent_1_low_item', '').lower() if pd.notna(row.get('mturk_agent_1_low_item', '')) else ''
            }
            
            agent2_prefs = {
                'high_item': row.get('mturk_agent_2_high_item', '').lower() if pd.notna(row.get('mturk_agent_2_high_item', '')) else '',
                'medium_item': row.get('mturk_agent_2_medium_item', '').lower() if pd.notna(row.get('mturk_agent_2_medium_item', '')) else '',
                'low_item': row.get('mturk_agent_2_low_item', '').lower() if pd.notna(row.get('mturk_agent_2_low_item', '')) else ''
            }
            
            # Convert preferences to dictionary format for utility calculation
            agent1_preference_dict = {}
            agent2_preference_dict = {}
            
            # Set preferences based on high/medium/low priority items
            for item in ['food', 'water', 'firewood']:
                if item == agent1_prefs['high_item'].lower():
                    agent1_preference_dict[item] = 5  # High priority: 5
                elif item == agent1_prefs['medium_item'].lower():
                    agent1_preference_dict[item] = 4  # Medium priority: 4
                elif item == agent1_prefs['low_item'].lower():
                    agent1_preference_dict[item] = 3  # Low priority: 3
                else:
                    # Fallback if preference mapping is incomplete
                    print(f"Warning: Incomplete preference mapping for agent1 in dialogue {dialogue_id}, item {item}")
                    agent1_preference_dict[item] = 4  # Default to medium priority
                    
                if item == agent2_prefs['high_item'].lower():
                    agent2_preference_dict[item] = 5  # High priority: 5
                elif item == agent2_prefs['medium_item'].lower():
                    agent2_preference_dict[item] = 4  # Medium priority: 4
                elif item == agent2_prefs['low_item'].lower():
                    agent2_preference_dict[item] = 3  # Low priority: 3
                else:
                    # Fallback if preference mapping is incomplete
                    print(f"Warning: Incomplete preference mapping for agent2 in dialogue {dialogue_id}, item {item}")
                    agent2_preference_dict[item] = 4  # Default to medium priority
            
            # Store preferences for this dialogue
            dialogue_preferences[dialogue_id] = {
                'agent1': agent1_preference_dict,
                'agent2': agent2_preference_dict
            }
            
            # Debug print for the first few dialogues
            if idx < 3:
                print(f"Dialogue {dialogue_id} preferences:")
                print(f"  Agent1: {agent1_preference_dict}")
                print(f"  Agent2: {agent2_preference_dict}")
        
        print(f"Loaded preferences for {len(dialogue_preferences)} dialogues with ratio {ratio}")
        if duplicate_count > 0:
            print(f"Warning: Found {duplicate_count} duplicate dialogue IDs (used first occurrence only)")
        
        return dialogue_preferences
    
    except Exception as e:
        print(f"Error loading preferences for ratio {ratio}: {e}")
        # Return a default preferences dict if loading fails
        return {}

# Initialize a dictionary to store preferences by ratio
DIALOGUE_PREFERENCES_BY_RATIO = {}

# Function to get preferences for a specific ratio
def get_preferences_for_ratio(ratio="0.5"):
    # If not loaded yet, load preferences for this ratio
    if ratio not in DIALOGUE_PREFERENCES_BY_RATIO:
        DIALOGUE_PREFERENCES_BY_RATIO[ratio] = load_preferences(ratio)
    return DIALOGUE_PREFERENCES_BY_RATIO[ratio]

# Function to calculate metrics for a single file - updated to use ratio parameter
def analyze_file(filepath):
    print(f"Processing {filepath}...")
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        print(f"Total rows in file: {len(df)}")
        
        # Extract ratio from filename
        ratio = extract_ratio(os.path.basename(filepath))
        print(f"Using ratio: {ratio}")
        
        # Get preferences for this ratio
        dialogue_preferences = get_preferences_for_ratio(ratio)
        
        # Filter out any rows with missing data
        valid_columns = [
            'agent1_food_actual', 'agent1_food_pred', 
            'agent1_water_actual', 'agent1_water_pred',
            'agent1_firewood_actual', 'agent1_firewood_pred',
            'agent2_food_actual', 'agent2_food_pred',
            'agent2_water_actual', 'agent2_water_pred',
            'agent2_firewood_actual', 'agent2_firewood_pred',
            'dialogue_id'
        ]
        
        # Create a copy of the dataframe to prevent SettingWithCopyWarning
        valid_data = df.dropna(subset=valid_columns).copy()
        print(f"Total rows in file after filtering: {len(valid_data)}")
        
        # Create a column for agent1 and agent2 utility using dynamic preferences
        valid_data.loc[:, 'agent1_utility_actual_calc'] = 0
        valid_data.loc[:, 'agent1_utility_pred_calc'] = 0
        valid_data.loc[:, 'agent2_utility_actual_calc'] = 0
        valid_data.loc[:, 'agent2_utility_pred_calc'] = 0
        
        # Process each row, using dialogue-specific preferences where available
        for idx, row in valid_data.iterrows():
            dialogue_id = row['dialogue_id']
            
            # Get preferences for this dialogue, or use defaults if not found
            if dialogue_id in dialogue_preferences:
                agent1_preferences = dialogue_preferences[dialogue_id]['agent1']
                agent2_preferences = dialogue_preferences[dialogue_id]['agent2']
            else:
                print(f"Warning: No preferences found for dialogue_id {dialogue_id}, using defaults")
                # Default preferences if not found
                agent1_preferences = {'food': 4, 'water': 4, 'firewood': 4}
                agent2_preferences = {'food': 4, 'water': 4, 'firewood': 4}
            
            # Calculate utilities using .loc instead of .at to avoid warnings
            valid_data.loc[idx, 'agent1_utility_actual_calc'] = calculate_utility(
                row['agent1_food_actual'],
                row['agent1_water_actual'],
                row['agent1_firewood_actual'],
                agent1_preferences
            )
            
            valid_data.loc[idx, 'agent2_utility_actual_calc'] = calculate_utility(
                row['agent2_food_actual'],
                row['agent2_water_actual'],
                row['agent2_firewood_actual'],
                agent2_preferences
            )
            
            # Calculate predicted utilities
            valid_data.loc[idx, 'agent1_utility_pred_calc'] = calculate_utility(
                row['agent1_food_pred'],
                row['agent1_water_pred'],
                row['agent1_firewood_pred'],
                agent1_preferences
            )
            
            valid_data.loc[idx, 'agent2_utility_pred_calc'] = calculate_utility(
                row['agent2_food_pred'],
                row['agent2_water_pred'],
                row['agent2_firewood_pred'],
                agent2_preferences
            )
        
        # Calculate MSE for agent utilities
        agent1_utility_mse = mean_squared_error(
            valid_data['agent1_utility_actual_calc'], 
            valid_data['agent1_utility_pred_calc']
        )
        
        agent2_utility_mse = mean_squared_error(
            valid_data['agent2_utility_actual_calc'], 
            valid_data['agent2_utility_pred_calc']
        )
        
        avg_utility_mse = (agent1_utility_mse + agent2_utility_mse) / 2
        
        # Calculate resource allocation match metrics
        food_match = (valid_data['agent1_food_actual'] == valid_data['agent1_food_pred']).mean()
        water_match = (valid_data['agent1_water_actual'] == valid_data['agent1_water_pred']).mean()
        firewood_match = (valid_data['agent1_firewood_actual'] == valid_data['agent1_firewood_pred']).mean()
        
        # Calculate exact match (all resources correctly predicted)
        exact_match = ((valid_data['agent1_food_actual'] == valid_data['agent1_food_pred']) & 
                        (valid_data['agent1_water_actual'] == valid_data['agent1_water_pred']) &
                        (valid_data['agent1_firewood_actual'] == valid_data['agent1_firewood_pred']) &
                        (valid_data['agent2_food_actual'] == valid_data['agent2_food_pred']) &
                        (valid_data['agent2_water_actual'] == valid_data['agent2_water_pred']) &
                        (valid_data['agent2_firewood_actual'] == valid_data['agent2_firewood_pred'])).mean()
        
        # Calculate additional metrics
        # Overall resource match (percentage of all resources correctly predicted)
        resource_match_overall = (
            (valid_data['agent1_food_actual'] == valid_data['agent1_food_pred']).sum() +
            (valid_data['agent1_water_actual'] == valid_data['agent1_water_pred']).sum() +
            (valid_data['agent1_firewood_actual'] == valid_data['agent1_firewood_pred']).sum() +
            (valid_data['agent2_food_actual'] == valid_data['agent2_food_pred']).sum() +
            (valid_data['agent2_water_actual'] == valid_data['agent2_water_pred']).sum() +
            (valid_data['agent2_firewood_actual'] == valid_data['agent2_firewood_pred']).sum()
        ) / (len(valid_data) * 6)  # 6 resources total
        
        return {
            'filename': os.path.basename(filepath),
            'ratio': ratio,
            'metrics': {
                'agent1_utility_mse': agent1_utility_mse,
                'agent2_utility_mse': agent2_utility_mse,
                'avg_utility_mse': avg_utility_mse,
                'food_match': food_match,
                'water_match': water_match,
                'firewood_match': firewood_match,
                'exact_match': exact_match,
                'resource_match_overall': resource_match_overall
            },
            'count': len(valid_data)
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

# Map file pattern to configuration type
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
            config_type = '(i) Utterance only'
        else:
            config_type = '(ii) Utterance + Intentions'
    elif summary_prefix == '(ii)':
        if 'no Intentions' in intentions_suffix:
            config_type = '(iii) Utterance + SCD'
        else:
            config_type = '(v) Utterance + SCD + Intentions'
    elif summary_prefix == '(iii)':
        if 'no Intentions' in intentions_suffix:
            config_type = '(iv) Utterance + SCM'
        else:
            config_type = '(vi) Utterance + SCM + Intentions'
    
    return config_type

# Extract model type from filename or path
def extract_model_type(filepath, filename):
    model_patterns = ['llama70b']
    for pattern in model_patterns:
        if pattern in filepath or pattern in filename:
            return pattern
    return 'unknown'

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
                result['filepath'] = file_path  # Store the full path
                results.append(result)
                print(f"Completed {file_path}: ", result.get('metrics', 'Error'))
            except Exception as e:
                print(f"Error with {file_path}: {e}")
        
        # Extract model types, config types, and ratios
        model_types = set()
        config_types = set()
        ratios = set()
        
        for result in results:
            if 'error' in result:
                continue
            
            filename = result['filename']
            filepath = result['filepath']
            ratio = result.get('ratio', '0.5')  # Default to 0.5 if not found
            
            model_type = extract_model_type(filepath, filename)
            config_type = map_file_to_config_type(filepath, filename)
            
            if model_type and config_type and config_type != 'Unknown':
                model_types.add(model_type)
                config_types.add(config_type)
                ratios.add(ratio)
        
        model_types = sorted(list(model_types))
        config_types = sorted(list(config_types), key=lambda x: x.split('(')[1].split(')')[0] if '(' in x else '')
        ratios = sorted(list(ratios), key=lambda x: float(x))
        
        print(f"Found {len(model_types)} model types: {model_types}")
        print(f"Found {len(config_types)} config types: {config_types}")
        print(f"Found {len(ratios)} ratios: {ratios}")
        
        # Check if we have standard ratios or need to use actual ones
        standard_ratios = ['0.25', '0.375', '0.5', '0.625', '0.75']
        if len(ratios) <= 1:
            print("Limited ratios found in filenames, using standard set")
            ratios = standard_ratios
        
        # Organize results for the table based on ratios
        metrics_tables = {
            'agent1_utility_mse': {},
            'agent2_utility_mse': {},
            'avg_utility_mse': {},
            'food_match': {},
            'water_match': {},
            'firewood_match': {},
            'exact_match': {},
            'resource_match_overall': {}
        }
        
        # Initialize table structure with ratios
        for metric_type in metrics_tables.keys():
            metrics_tables[metric_type] = {}
            
            for config_type in config_types:
                metrics_tables[metric_type][config_type] = {}
                
                for ratio in ratios:
                    metrics_tables[metric_type][config_type][ratio] = {}
                    
                    for model_type in model_types:
                        metrics_tables[metric_type][config_type][ratio][model_type] = None
        
        # Fill in table data
        for result in results:
            if 'error' in result:
                continue
            
            filename = result['filename']
            filepath = result['filepath']
            ratio = result.get('ratio', '0.5')  # Default to 0.5 if not found
            
            model_type = extract_model_type(filepath, filename)
            config_type = map_file_to_config_type(filepath, filename)
            
            if not model_type or not config_type or config_type == 'Unknown':
                continue
            
            # Skip if ratio not in our selected list
            if ratio not in ratios:
                continue
            
            for metric in metrics_tables.keys():
                # Make sure all nested dictionaries exist
                if config_type not in metrics_tables[metric]:
                    metrics_tables[metric][config_type] = {}
                if ratio not in metrics_tables[metric][config_type]:
                    metrics_tables[metric][config_type][ratio] = {}
                
                metrics_tables[metric][config_type][ratio][model_type] = result['metrics'][metric]
        
        # Generate LaTeX table by ratio
        latex_table = generate_latex_table_by_ratio(metrics_tables, model_types, config_types, ratios)
        
        # Determine output file path
        output_file = os.path.join(base_dir, 'casino_metrics_ratio_table.tex')
        
        # Write LaTeX table to file
        with open(output_file, 'w') as f:
            f.write(latex_table)
        
        print(f"LaTeX table saved to {output_file}")
        return latex_table
    except Exception as e:
        print(f"Error processing files: {e}")
        return {'error': str(e)}

# Add this function to the code
def print_file_to_config_mapping(base_dir):
    """Print mapping of file paths to their configuration categories"""
    print("\n=== FILE PATH TO CONFIGURATION MAPPING ===\n")
    
    # Find all CSV files in the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'final_results' in file:
                filepath = os.path.join(root, file)
                filename = os.path.basename(filepath)
                
                # Map file to configuration type
                config_type = map_file_to_config_type(filepath, filename)
                
                # Print mapping
                print(f"{filepath}: {config_type}")
    
    print("\n=== END OF MAPPING ===\n")

# Generate LaTeX table organized by ratio like in paste2.txt
def generate_latex_table_by_ratio(metrics_tables, model_types, config_types, ratios):
    # Table for MSE metrics (lower is better)
    mse_metrics = ['agent1_utility_mse', 'agent2_utility_mse', 'avg_utility_mse']
    mse_metric_names = ['Agent 1 Utility MSE', 'Agent 2 Utility MSE', 'Average Utility MSE']
    
    # Table for match metrics (higher is better)
    match_metrics = ['food_match', 'water_match', 'firewood_match', 'exact_match', 'resource_match_overall']
    match_metric_names = ['Food Match Ratio', 'Water Match Ratio', 'Firewood Match Ratio', 'Exact Match Ratio', 'Overall Resource Match']
    
    # Combine all metrics
    all_metrics = mse_metrics + match_metrics
    all_metric_names = mse_metric_names + match_metric_names
    
    # Start the combined table
    latex_table = "\\begin{table*}[ht]\n\\centering\n"
    latex_table += "\\begin{tabular}{ll" + "c" * len(ratios) + "}\n\\hline\n"
    
    # Table header with ratio percentages
    latex_table += "\\textbf{Metric} & \\textbf{Config} & " + " & ".join([f"\\textbf{{{float(r) * 100:.1f}\\%}}" for r in ratios]) + " \\\\\n\\hline\n"
    
    # Get baseline config type for color coding
    baseline_type = next((ct for ct in config_types if "(i)" in ct), config_types[0])
    
    # For each metric and model type combination
    for model_type in model_types:
        # Add model type header
        latex_table += "\\multicolumn{" + str(len(ratios) + 2) + "}{l}{\\textbf{Model: " + model_type.upper() + "}} \\\\\n\\midrule\n"
        
        # For each metric
        for i, metric_name in enumerate(all_metric_names):
            metric_key = all_metrics[i]
            is_mse_metric = i < len(mse_metrics)  # True for MSE metrics, False for match metrics
            
            # Add proper superscript to first MSE and first match metric
            display_metric_name = metric_name
            if i == 0:
                display_metric_name = f"{metric_name}"
            elif i == len(mse_metrics):
                display_metric_name = f"{metric_name}"
            
            # Add metric name to first row only
            first_config = True
            
            # Get baseline values for each ratio for color coding
            baseline_values = {}
            for ratio in ratios:
                if baseline_type in metrics_tables[metric_key] and \
                   ratio in metrics_tables[metric_key][baseline_type] and \
                   model_type in metrics_tables[metric_key][baseline_type][ratio] and \
                   metrics_tables[metric_key][baseline_type][ratio][model_type] is not None:
                    baseline_values[ratio] = metrics_tables[metric_key][baseline_type][ratio][model_type]
            
            # Add rows for each config
            for config in config_types:
                row = ""
                
                # Add metric name only to first row of the metric section
                if first_config:
                    row += f"\\multirow{{{len(config_types)}}}{{*}}{{\\textbf{{{display_metric_name}}}}} & {config} & "
                    first_config = False
                else:
                    row += f" & {config} & "
                
                cells = []
                for ratio in ratios:
                    # Try to get the value for this config, ratio, and model
                    value = None
                    if config in metrics_tables[metric_key] and \
                       ratio in metrics_tables[metric_key][config] and \
                       model_type in metrics_tables[metric_key][config][ratio]:
                        value = metrics_tables[metric_key][config][ratio][model_type]
                    
                    if value is None:
                        cells.append('-')
                    else:
                        # Format value based on metric type
                        if is_mse_metric:
                            # Format MSE value (round to 4 decimal places)
                            formatted_value = f"{value:.4f}"
                            
                            # Color compared to baseline for MSE (lower is better)
                            if config == baseline_type or ratio not in baseline_values:
                                cells.append(formatted_value)
                            else:
                                is_better = value < baseline_values[ratio]
                                color_cmd = '\\cellcolor{green!25}' if is_better else '\\cellcolor{red!25}'
                                cells.append(f"{color_cmd}{formatted_value}")
                        else:
                            # Format match ratio as percentage (multiply by 100 and round to 2 decimal places)
                            formatted_value = f"{value * 100:.2f}\\%"
                            
                            # Color compared to baseline for match ratios (higher is better)
                            if config == baseline_type or ratio not in baseline_values:
                                cells.append(formatted_value)
                            else:
                                is_better = value > baseline_values[ratio]
                                color_cmd = '\\cellcolor{green!25}' if is_better else '\\cellcolor{red!25}'
                                cells.append(f"{color_cmd}{formatted_value}")
                
                row += " & ".join(cells) + " \\\\\n"
                latex_table += row
            
            # Add a midrule between metrics (except after the last one)
            if i < len(all_metric_names) - 1:
                latex_table += "\\midrule\n"
            
        # Add a separator between model types (except after the last one)
        if model_type != model_types[-1]:
            latex_table += "\\midrule\n"
    
    # Add footnotes for the metric types
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    
    # Add final caption
    latex_table += "\\caption{\\textbf{ICL: }Performance metrics across different conversation ratios and configuration types for LLAMA-3.1-70B on the \\textbf{Casino} dataset}\n"
    latex_table += "\\label{tab:combined_metrics_icl_llama_casino_by_ratio}\n\\end{table*}"
    
    return latex_table

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Update paths from arguments
    base_dir = args.results_dir
    preferences_csv_path = args.preferences_file
    
    print("hi")
    # Print file to config mapping
    print_file_to_config_mapping(base_dir)

    # Process files and generate LaTeX table
    latex_table = process_all_files()
    
    # Determine output file path
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(base_dir, 'casino_metrics_ratio_table.tex')
    
    # Write LaTeX table to file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")