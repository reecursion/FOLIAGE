import os
import re
import csv
import pandas as pd
from pathlib import Path

def extract_allocation(response):
    """Extract final allocation from model output for Casino dataset."""
    try:
        # Print a sample for debugging the first time
        if not hasattr(extract_allocation, "debug_printed"):
            print(f"DEBUG: Sample text to extract from:\n{response}...")
            extract_allocation.debug_printed = True
        
        # Look directly for OUTCOME pattern which is the most reliable
        outcome_match = re.search(r'OUTCOME:\s*({agent1:{food:\d+,\s*water:\d+,\s*firewood:\d+},\s*agent2:{food:\d+,\s*water:\d+,\s*firewood:\d+}})', response)
        if outcome_match:
            outcome_text = outcome_match.group(1)
            # print(f"DEBUG: Found outcome text: {outcome_text}")
        else:
            # Try other preprocessing if direct match fails
            if "<|end_header_id|>" in response:
                parts = response.split("<|end_header_id|>")
                for part in parts:
                    if "OUTCOME:" in part:
                        response = part
                        break
            
            # Try looking for the pattern after cleaning up more tokens
            response = re.sub(r'<\|[^>]+\|>', '', response)
            
            # Sometimes the format is buried in the text
            outcome_match = re.search(r'OUTCOME:\s*({agent1:{food:\d+,\s*water:\d+,\s*firewood:\d+},\s*agent2:{food:\d+,\s*water:\d+,\s*firewood:\d+}})', response)
            if outcome_match:
                outcome_text = outcome_match.group(1)
                print(f"DEBUG: Found outcome text after cleaning: {outcome_text}")
            else:
                # Final fallback to find any matching pattern anywhere in the text
                pattern = r'{agent1:{food:(\d+),\s*water:(\d+),\s*firewood:(\d+)},\s*agent2:{food:(\d+),\s*water:(\d+),\s*firewood:(\d+)}}'
                braces_match = re.search(pattern, response)
                if braces_match:
                    allocation = {'agent1': {}, 'agent2': {}}
                    allocation['agent1']['food'] = int(braces_match.group(1))
                    allocation['agent1']['water'] = int(braces_match.group(2))
                    allocation['agent1']['firewood'] = int(braces_match.group(3))
                    allocation['agent2']['food'] = int(braces_match.group(4))
                    allocation['agent2']['water'] = int(braces_match.group(5))
                    allocation['agent2']['firewood'] = int(braces_match.group(6))
                    return allocation
                else:
                    return None
        
        # Process the found outcome text
        allocation = {'agent1': {}, 'agent2': {}}
        # Pattern for the curly braces format
        pattern = r'{agent1:{food:(\d+),\s*water:(\d+),\s*firewood:(\d+)},\s*agent2:{food:(\d+),\s*water:(\d+),\s*firewood:(\d+)}}'
        braces_match = re.search(pattern, outcome_text)
        
        if braces_match:
            # Extract values from regex groups
            allocation['agent1']['food'] = int(braces_match.group(1))
            allocation['agent1']['water'] = int(braces_match.group(2))
            allocation['agent1']['firewood'] = int(braces_match.group(3))
            allocation['agent2']['food'] = int(braces_match.group(4))
            allocation['agent2']['water'] = int(braces_match.group(5))
            allocation['agent2']['firewood'] = int(braces_match.group(6))
            return allocation
        
        return None
    except Exception as e:
        print(f"[ERROR] Failed to extract allocation: {e}")
        return None

def calculate_utility_score(allocation, preferences):
    """Calculate utility score based on allocation and preferences."""
    utility_map = {
        'high': 5,
        'medium': 4,
        'low': 3
    }
    
    score = 0
    for agent, items in allocation.items():
        for item, value in items.items():
            # Get preference level for this agent and item
            pref = preferences[agent][item]
            score += utility_map[pref] * value
    
    return score

def extract_preferences(prompt):
    """Extract agent preferences from the prompt."""
    preferences = {
        'agent1': {'food': None, 'water': None, 'firewood': None},
        'agent2': {'food': None, 'water': None, 'firewood': None}
    }
    
    # Extract Agent 1 preferences
    agent1_match = re.search(r'Agent 1: High priority: (\w+), Medium priority: (\w+), Low priority: (\w+)', prompt)
    if agent1_match:
        preferences['agent1'][agent1_match.group(1)] = 'high'
        preferences['agent1'][agent1_match.group(2)] = 'medium'
        preferences['agent1'][agent1_match.group(3)] = 'low'
    
    # Extract Agent 2 preferences
    agent2_match = re.search(r'Agent 2: High priority: (\w+), Medium priority: (\w+), Low priority: (\w+)', prompt)
    if agent2_match:
        preferences['agent2'][agent2_match.group(1)] = 'high'
        preferences['agent2'][agent2_match.group(2)] = 'medium'
        preferences['agent2'][agent2_match.group(3)] = 'low'
    
    return preferences

def process_csv_file(csv_path):
    """Process a single CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        original_rows = len(df)
        print(f"\nProcessing {csv_path} ({original_rows} rows)")
        
        # Initialize counters
        valid_extractions = 0
        invalid_extractions = 0
        
        # Process each row
        for index, row in df.iterrows():
            prompt = row.get('prompt', '')
            generated_text = row.get('generated_text', '')
            
            # Extract allocation from generated text
            predicted_allocation = extract_allocation(generated_text)
            
            # If extraction is successful, calculate utilities and MSE
            if predicted_allocation is not None:
                valid_extractions += 1
                
                # Update predicted allocation in the dataframe
                df.at[index, 'predicted_agent1_food'] = predicted_allocation['agent1']['food']
                df.at[index, 'predicted_agent1_water'] = predicted_allocation['agent1']['water']
                df.at[index, 'predicted_agent1_firewood'] = predicted_allocation['agent1']['firewood']
                df.at[index, 'predicted_agent2_food'] = predicted_allocation['agent2']['food']
                df.at[index, 'predicted_agent2_water'] = predicted_allocation['agent2']['water']
                df.at[index, 'predicted_agent2_firewood'] = predicted_allocation['agent2']['firewood']
                
                # Extract preferences from prompt
                preferences = extract_preferences(prompt)
                
                # Calculate predicted utility scores
                agent1_utility = calculate_utility_score({'agent1': predicted_allocation['agent1']}, preferences)
                agent2_utility = calculate_utility_score({'agent2': predicted_allocation['agent2']}, preferences)
                
                # Get true utility scores from the dataframe
                true_agent1_utility = row.get('true_agent1_utility')
                true_agent2_utility = row.get('true_agent2_utility')
                
                # Update predicted utility in the dataframe
                df.at[index, 'predicted_agent1_utility'] = agent1_utility
                df.at[index, 'predicted_agent2_utility'] = agent2_utility
                
                # Calculate and update MSE if true utilities are available
                if pd.notnull(true_agent1_utility) and pd.notnull(true_agent2_utility):
                    utility_mse = ((true_agent1_utility - agent1_utility) ** 2 + 
                                  (true_agent2_utility - agent2_utility) ** 2) / 2
                    df.at[index, 'utility_mse'] = utility_mse
            else:
                invalid_extractions += 1
                # Leave predicted fields empty for invalid extractions
        
        # Write the updated dataframe back to the CSV
        df.to_csv(csv_path, index=False)
        
        # Print statistics
        print(f"  Total rows: {original_rows}")
        print(f"  Valid extractions: {valid_extractions}")
        print(f"  Invalid extractions: {invalid_extractions}")
        print(f"  Valid extraction rate: {valid_extractions/original_rows:.2%}")
        
        return {
            'total': original_rows,
            'valid': valid_extractions,
            'invalid': invalid_extractions
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")
        return {
            'total': 0,
            'valid': 0,
            'invalid': 0
        }

def process_single_row(row_data, verbose=False):
    """Process a single row for debugging purposes"""
    response = row_data.get('generated_text', '')
    print(f"\nDEBUG: Testing extraction on sample row")
    print(f"Generated text sample: {response[:100]}...")
    allocation = extract_allocation(response)
    if allocation:
        print(f"SUCCESS: Extracted allocation: {allocation}")
        return True
    else:
        print(f"FAILURE: Could not extract allocation")
        if verbose:
            print(f"Full text:\n{response}")
        return False

def crawl_directories(root_dir, debug_mode=True):
    """Crawl through directories to find and process CSV files."""
    total_stats = {
        'files_processed': 0,
        'total_rows': 0,
        'valid_extractions': 0,
        'invalid_extractions': 0
    }
    
    # Find all CSV files in the directory and subdirectories
    csv_files = list(Path(root_dir).rglob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    # Debug mode - check a sample row from the first file
    if debug_mode and csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            if len(df) > 0:
                sample_row = df.iloc[0].to_dict()
                process_single_row(sample_row, verbose=True)
        except Exception as e:
            print(f"DEBUG: Error during sample processing: {e}")
    
    # Process each CSV file
    for csv_file in csv_files:
        stats = process_csv_file(csv_file)
        
        # Update total statistics
        total_stats['files_processed'] += 1
        total_stats['total_rows'] += stats['total']
        total_stats['valid_extractions'] += stats['valid']
        total_stats['invalid_extractions'] += stats['invalid']
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"  Files processed: {total_stats['files_processed']}")
    print(f"  Total rows: {total_stats['total_rows']}")
    print(f"  Valid extractions: {total_stats['valid_extractions']}")
    print(f"  Invalid extractions: {total_stats['invalid_extractions']}")
    
    if total_stats['total_rows'] > 0:
        print(f"  Overall valid extraction rate: {total_stats['valid_extractions']/total_stats['total_rows']:.2%}")

if __name__ == "__main__":
    import sys
    
    # Check for a test sample first
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Sample text from your data
        sample_text = """<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

You are helping analyze a negotiation conversation where two agents are discussing the allocation of resources. Each resource has exactly three units that must be divided between the two agents. The total amount of each resource allocated to both agents must add up to three. 

            Agent Preferences:
            Agent 1: High priority: firewood, Medium priority: water, Low priority: food
            Agent 2: High priority: food, Medium priority: water, Low priority: firewood

            Conversation with intentions:
            Agent 2: Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop., Agent 1: oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold., [Summary: The conversation begins with one speaker expressing a need for food due to a health condition, while the other empathizes and shares a personal connection to the issue. The second speaker then proposes a trade, indicating a willingness to negotiate based on mutual health-related needs. Both speakers maintain a cooperative tone, focusing on finding a solution that addresses their respective challenges.]

            Based on this negotiation, predict the final allocation of resources. Provide your answer using the following format with curly braces, with no explanation:

            OUTCOME: {agent1:{food:[number], water:[number], firewood:[number]}, agent2:{food:[number], water:[number], firewood:[number]}}

            Remember: Each resource must sum to exactly 3 units across both agents.<|eot_id|><|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

OUTCOME: {agent1:{food:0, water:2, firewood:3}, agent2:{food:3, water:1, firewood:0}}<|eot_id|>"""
        
        result = extract_allocation(sample_text)
        if result:
            print(f"TEST PASSED! Extracted: {result}")
        else:
            print("TEST FAILED! Could not extract the allocation.")
            
        # Alternative format test
        alt_sample = """OUTCOME: {agent1:{food:0, water:2, firewood:3}, agent2:{food:3, water:1, firewood:0}}"""
        result = extract_allocation(alt_sample)
        if result:
            print(f"ALTERNATIVE TEST PASSED! Extracted: {result}")
        else:
            print("ALTERNATIVE TEST FAILED! Could not extract the allocation.")
        
        sys.exit(0)
    
    # Normal operation with directory processing
    if len(sys.argv) > 1 and sys.argv[1] != '--test':
        root_directory = sys.argv[1]
    else:
        root_directory = input("Enter the root directory to search for CSV files: ")
    
    crawl_directories(root_directory)