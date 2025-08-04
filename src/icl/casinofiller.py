import os
import re
import pandas as pd

class CasinoFiller:
    def __init__(self, csv_dir, dataset_path):
        self.csv_dir = csv_dir
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        self.dataset_type = "casino"

    def extract_allocation(self, response):
        ALLOCATION_PATTERN = re.compile(
            r'{agent1:{food:\[?(\d+)\]?,\s*water:\[?(\d+)\]?,\s*firewood:\[?(\d+)\]?},\s*agent2:{food:\[?(\d+)\]?,\s*water:\[?(\d+)\]?,\s*firewood:\[?(\d+)\]?}}',
            re.IGNORECASE
        )
        try:
            if not response or not isinstance(response, str):
                return None
            braces_match = ALLOCATION_PATTERN.search(response)
            if not braces_match:
                for token in ["<|end_header_id|>", "assistant", "OUTCOME"]:
                    if token in response:
                        response = response.split(token, 1)[-1].strip()
                braces_match = ALLOCATION_PATTERN.search(response)
            if braces_match:
                return {
                    'agent1': {
                        'food': int(braces_match.group(1)),
                        'water': int(braces_match.group(2)),
                        'firewood': int(braces_match.group(3))
                    },
                    'agent2': {
                        'food': int(braces_match.group(4)),
                        'water': int(braces_match.group(5)),
                        'firewood': int(braces_match.group(6))
                    }
                }
            return None
        except Exception as e:
            print(f"[ERROR] extract_allocation: {e}")
            return None

    def calculate_utility_score(self, allocation, preferences):
        utility_map = {'high': 5, 'medium': 4, 'low': 3}
        score = 0
        for agent, items in allocation.items():
            for item, value in items.items():
                pref = preferences[agent][item]
                score += utility_map[pref] * value
        return score

    def determine_actual_outcome(self, row):
        try:
            actual_allocation = {
                'agent1': {
                    'food': int(row.get('mturk_agent_1_food', 0)),
                    'water': int(row.get('mturk_agent_1_water', 0)),
                    'firewood': int(row.get('mturk_agent_1_firewood', 0))
                },
                'agent2': {
                    'food': int(row.get('mturk_agent_2_food', 0)),
                    'water': int(row.get('mturk_agent_2_water', 0)),
                    'firewood': int(row.get('mturk_agent_2_firewood', 0))
                }
            }
            preferences = {
                'agent1': {
                    row.get('mturk_agent_1_high_item', '').lower(): 'high',
                    row.get('mturk_agent_1_medium_item', '').lower(): 'medium',
                    row.get('mturk_agent_1_low_item', '').lower(): 'low'
                },
                'agent2': {
                    row.get('mturk_agent_2_high_item', '').lower(): 'high',
                    row.get('mturk_agent_2_medium_item', '').lower(): 'medium',
                    row.get('mturk_agent_2_low_item', '').lower(): 'low'
                }
            }
            agent1_utility = self.calculate_utility_score({'agent1': actual_allocation['agent1']}, preferences)
            agent2_utility = self.calculate_utility_score({'agent2': actual_allocation['agent2']}, preferences)
            return {
                'allocation': actual_allocation,
                'preferences': preferences,
                'agent1_utility': agent1_utility,
                'agent2_utility': agent2_utility
            }
        except Exception as e:
            print(f"[WARNING] determine_actual_outcome: {e}")
            return None

    def process_files(self):
        for root, _, files in os.walk(self.csv_dir):
            for filename in files:
                if filename.endswith(".csv"):
                    filepath = os.path.join(root, filename)
                    df = pd.read_csv(filepath)

                    for i, row in df.iterrows():
                        if pd.isna(row['agent1_food_pred']):
                            allocation = self.extract_allocation(row['response'])
                            if allocation:
                                df.at[i, 'agent1_food_pred'] = allocation['agent1']['food']
                                df.at[i, 'agent1_water_pred'] = allocation['agent1']['water']
                                df.at[i, 'agent1_firewood_pred'] = allocation['agent1']['firewood']
                                df.at[i, 'agent2_food_pred'] = allocation['agent2']['food']
                                df.at[i, 'agent2_water_pred'] = allocation['agent2']['water']
                                df.at[i, 'agent2_firewood_pred'] = allocation['agent2']['firewood']

                        if pd.isna(row['agent1_utility_pred']) or pd.isna(row['utility_mse']):
                            actual_row = self.dataset[self.dataset['dialogue_id'] == row['dialogue_id']]
                            if actual_row.empty:
                                continue
                            actual_row = actual_row.iloc[0]
                            actual_outcome = self.determine_actual_outcome(actual_row)
                            if actual_outcome and allocation:
                                preferences = actual_outcome['preferences']
                                agent1_utility_pred = self.calculate_utility_score({'agent1': allocation['agent1']}, preferences)
                                agent2_utility_pred = self.calculate_utility_score({'agent2': allocation['agent2']}, preferences)
                                
                                # Store actual allocations
                                alloc = actual_outcome['allocation']
                                df.at[i, 'agent1_food_actual'] = alloc['agent1']['food']
                                df.at[i, 'agent1_water_actual'] = alloc['agent1']['water']
                                df.at[i, 'agent1_firewood_actual'] = alloc['agent1']['firewood']
                                df.at[i, 'agent2_food_actual'] = alloc['agent2']['food']
                                df.at[i, 'agent2_water_actual'] = alloc['agent2']['water']
                                df.at[i, 'agent2_firewood_actual'] = alloc['agent2']['firewood']

                                df.at[i, 'agent1_utility_pred'] = agent1_utility_pred
                                df.at[i, 'agent2_utility_pred'] = agent2_utility_pred
                                df.at[i, 'agent1_utility_actual'] = actual_outcome['agent1_utility']
                                df.at[i, 'agent2_utility_actual'] = actual_outcome['agent2_utility']
                                df.at[i, 'utility_mse'] = (actual_outcome['agent1_utility'] - agent1_utility_pred) ** 2

                    df.to_csv(filepath, index=False)
                    print(f"[INFO] Updated: {filename}")

# Example usage
csv_dir = "/Users/rithviksenthil/Desktop/FOLIAGE/src/icl/results/casino/"
dataset_path = "/Users/rithviksenthil/Desktop/FOLIAGE/datasets/casino/final/ratio_0.5.csv"
filler = CasinoFiller(csv_dir, dataset_path)
filler.process_files()
