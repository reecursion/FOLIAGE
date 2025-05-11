import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error

# Path to the directory containing the CSV files
base_dir = "/home/rithviks/FOLIAGE/src/sft/results/casino"  

preferences_csv_path = "/home/rithviks/FOLIAGE/datasets/casino/final/ratio_0.5.csv"

# Command line argument parsing
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process casino dataset metrics.')
    parser.add_argument('--results_dir', type=str, default="/home/rithviks/FOLIAGE/src/sft/results/casino",
                        help='Directory containing result CSV files')
    parser.add_argument('--preferences_file', type=str, 
                        default="/home/rithviks/FOLIAGE/datasets/casino/final/ratio_0.5.csv",
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


def load_preferences(ratio="0.5"):
    ratio_preferences_path = preferences_csv_path.replace("ratio_0.5", f"ratio_{ratio}")
    print(f"Loading preferences data from {ratio_preferences_path}")
    
    try:
        prefs_df = pd.read_csv(ratio_preferences_path)
        dialogue_preferences = {}
        duplicate_count = 0
        
        for idx, row in prefs_df.iterrows():
            dialogue_id = row['dialogue_id']
            
            if dialogue_id in dialogue_preferences:
                duplicate_count += 1
                # print(f"Warning: Duplicate dialogue_id {dialogue_id} found (row {idx}). Using first occurrence only.")
                continue
            
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
            
            agent1_preference_dict = {}
            agent2_preference_dict = {}
            
            for item in ['food', 'water', 'firewood']:
                if item == agent1_prefs['high_item'].lower():
                    agent1_preference_dict[item] = 5  # High priority: 5
                elif item == agent1_prefs['medium_item'].lower():
                    agent1_preference_dict[item] = 4  # Medium priority: 4
                elif item == agent1_prefs['low_item'].lower():
                    agent1_preference_dict[item] = 3  # Low priority: 3
                else:
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
        return {}

DIALOGUE_PREFERENCES_BY_RATIO = {}

def get_preferences_for_ratio(ratio="0.5"):
    if ratio not in DIALOGUE_PREFERENCES_BY_RATIO:
        DIALOGUE_PREFERENCES_BY_RATIO[ratio] = load_preferences(ratio)
    return DIALOGUE_PREFERENCES_BY_RATIO[ratio]

def analyze_file(filepath):
    print(f"Processing {filepath}...")
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        print(f"Total rows in file: {len(df)}")
        
        ratio = extract_ratio(os.path.basename(filepath))
        print(f"Using ratio: {ratio}")
        
        dialogue_preferences = get_preferences_for_ratio(ratio)
        
        valid_columns = [
            'agent1_food', 'predicted_agent1_food', 
            'agent1_water', 'predicted_agent1_water',
            'agent1_firewood', 'predicted_agent1_firewood',
            'agent2_food', 'predicted_agent2_food',
            'agent2_water', 'predicted_agent2_water',
            'agent2_firewood', 'predicted_agent2_firewood',
            'dialogue_id'
        ]
        
        valid_data = df.dropna(subset=valid_columns).copy()
        print(f"Total rows in file after filtering: {len(valid_data)}")
        
        valid_data.loc[:, 'agent1_utility_actual_calc'] = 0
        valid_data.loc[:, 'agent1_utility_pred_calc'] = 0
        valid_data.loc[:, 'agent2_utility_actual_calc'] = 0
        valid_data.loc[:, 'agent2_utility_pred_calc'] = 0
        
        for idx, row in valid_data.iterrows():
            dialogue_id = row['dialogue_id']
            
            if dialogue_id in dialogue_preferences:
                agent1_preferences = dialogue_preferences[dialogue_id]['agent1']
                agent2_preferences = dialogue_preferences[dialogue_id]['agent2']
            else:
                print(f"Warning: No preferences found for dialogue_id {dialogue_id}, using defaults")
                # Default preferences if not found
                agent1_preferences = {'food': 4, 'water': 4, 'firewood': 4}
                agent2_preferences = {'food': 4, 'water': 4, 'firewood': 4}
            
            valid_data.loc[idx, 'agent1_utility_actual_calc'] = calculate_utility(
                row['agent1_food'],
                row['agent1_water'],
                row['agent1_firewood'],
                agent1_preferences
            )
            
            valid_data.loc[idx, 'agent2_utility_actual_calc'] = calculate_utility(
                row['agent2_food'],
                row['agent2_water'],
                row['agent2_firewood'],
                agent2_preferences
            )
            
            # Calculate predicted utilities
            valid_data.loc[idx, 'agent1_utility_pred_calc'] = calculate_utility(
                row['predicted_agent1_food'],
                row['predicted_agent1_water'],
                row['predicted_agent1_firewood'],
                agent1_preferences
            )
            
            valid_data.loc[idx, 'agent2_utility_pred_calc'] = calculate_utility(
                row['predicted_agent2_food'],
                row['predicted_agent2_water'],
                row['predicted_agent2_firewood'],
                agent2_preferences
            )
        
        agent1_utility_mse = mean_squared_error(
            valid_data['agent1_utility_actual_calc'], 
            valid_data['agent1_utility_pred_calc']
        )
        
        agent2_utility_mse = mean_squared_error(
            valid_data['agent2_utility_actual_calc'], 
            valid_data['agent2_utility_pred_calc']
        )
        
        avg_utility_mse = (agent1_utility_mse + agent2_utility_mse) / 2
        
        food_match = (valid_data['agent1_food'] == valid_data['predicted_agent1_food']).mean()
        water_match = (valid_data['agent1_water'] == valid_data['predicted_agent1_water']).mean()
        firewood_match = (valid_data['agent1_firewood'] == valid_data['predicted_agent1_firewood']).mean()
        
        exact_match = ((valid_data['agent1_food'] == valid_data['predicted_agent1_food']) & 
                        (valid_data['agent1_water'] == valid_data['predicted_agent1_water']) &
                        (valid_data['agent1_firewood'] == valid_data['predicted_agent1_firewood']) &
                        (valid_data['agent2_food'] == valid_data['predicted_agent2_food']) &
                        (valid_data['agent2_water'] == valid_data['predicted_agent2_water']) &
                        (valid_data['agent2_firewood'] == valid_data['predicted_agent2_firewood'])).mean()
        
        resource_match_overall = (
            (valid_data['agent1_food'] == valid_data['predicted_agent1_food']).sum() +
            (valid_data['agent1_water'] == valid_data['predicted_agent1_water']).sum() +
            (valid_data['agent1_firewood'] == valid_data['predicted_agent1_firewood']).sum() +
            (valid_data['agent2_food'] == valid_data['predicted_agent2_food']).sum() +
            (valid_data['agent2_water'] == valid_data['predicted_agent2_water']).sum() +
            (valid_data['agent2_firewood'] == valid_data['predicted_agent2_firewood']).sum()
        ) / (len(valid_data) * 6)  
        
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
    match = re.search(r'casino_ratio_(\d+\.\d+)', filename)
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
                print(f"Completed {file_path}: ", result.get('metrics', 'Error'))
            except Exception as e:
                print(f"Error with {file_path}: {e}")
        
        # Organize results for the table
        table_data = {
            'agent1_utility_mse': {},
            'agent2_utility_mse': {},
            'avg_utility_mse': {},
            'food_match': {},
            'water_match': {},
            'firewood_match': {},
            'exact_match': {},
            'resource_match_overall': {}
        }
        
        # Initialize structure
        config_types = [
            '(i) Utterance',
            '(ii) Utterance + Intentions',
            '(iii) Utterance + SCD Summary',
            '(iv) Utterance + SCM Summary',
            '(v) Utterance + Traditional Summary',
            '(vi) Utt + Intentions + SCD Summary',
            '(vii) Utt + Intentions + SCM Summary',
            '(viii) Utt + Intentions + Traditional Summary'
        ]
        
        ratios = ['0.25', '0.375', '0.5', '0.625', '0.75']
        
        # Initialize empty table
        for metric_type in ['agent1_utility_mse','agent2_utility_mse','avg_utility_mse','food_match',
            'water_match', 'firewood_match', 'exact_match', 'resource_match_overall']:
            table_data[metric_type] = {}
            
            for config_type in config_types:
                table_data[metric_type][config_type] = {}
                
                for ratio in ratios:
                    table_data[metric_type][config_type][ratio] = 0
        
        # Fill in table data
        for result in results:
            if 'error' in result:
                continue
            
            config_type = map_file_to_config_type(result['filename'])
            ratio = extract_ratio(result['filename'])
            
            if not config_type or not ratio or ratio not in ratios:
                continue
            avg_metrics = result['metrics']
            for metric in ['agent1_utility_mse','agent2_utility_mse','avg_utility_mse','food_match',
            'water_match', 'firewood_match', 'exact_match', 'resource_match_overall']:
                if metric in avg_metrics:
                    table_data[metric][config_type][ratio] += avg_metrics[metric]
        
        for metric in table_data:
            for config_type in table_data[metric]:
                for ratio in table_data[metric][config_type]:
                    table_data[metric][config_type][ratio] /= 3

        # Generate combined LaTeX table
        print("\n--- Combined Table for All Metrics ---")
        generate_combined_latex_table(table_data, ratios)
        
        return results
    except Exception as e:
        print(f"Error processing files: {e}")
        return {'error': str(e)}

# Generate combined LaTeX table for all metrics with standard deviations
def generate_combined_latex_table(data, ratios):
    metric_names = ['Agent 1 Utility MSE', 'Agent 2 Utility MSE', 'Average Utility MSE', 'Food Match Ratio', 'Water Match Ratio', 'Firewood Match Ratio', 'Exact Match Ratio', 'Overall Resource Match']
    metric_keys = ['agent1_utility_mse', 'agent2_utility_mse', 'avg_utility_mse', 'food_match', 'water_match', 'firewood_match', 'exact_match', 'resource_match_overall']
    
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

if __name__ == "__main__":
    process_all_files()
