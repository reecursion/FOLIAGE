import pandas as pd
import os
import json
import argparse
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from tqdm import tqdm
import re

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

class ConversationForecastingWithIntentions:
    def __init__(self, dataset_type, model_type="gpt", summary_type="none", 
                 ratio=0.5, batch_size=5, include_intentions=False):
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.summary_type = summary_type
        self.ratio = ratio
        self.batch_size = batch_size
        self.include_intentions = include_intentions
        
        if self.dataset_type == "cb":
            self.dataset_dir = "/home/gganeshl/FOLIAGE/datasets/craigslistbargain/final"
        elif self.dataset_type == "p4g":
            self.dataset_dir = "/home/gganeshl/FOLIAGE/datasets/p4g/final"
        elif self.dataset_type == "casino":
            self.dataset_dir = "/home/gganeshl/FOLIAGE/datasets/casino/final"
        
        # Base output directory
        base_output_dir = f"/home/gganeshl/FOLIAGE/src/icl/results/{dataset_type}/{model_type}/seed_{args.seed}"
        
        # Determine the appropriate subdirectory based on intentions and summary type
        if self.include_intentions and self.summary_type == "none":
            self.output_dir = os.path.join(base_output_dir, "localscaffolding")
        elif not self.include_intentions and self.summary_type != "none":
            self.output_dir = os.path.join(base_output_dir, "globalscaffolding")
        elif self.include_intentions and self.summary_type != "none":
            self.output_dir = os.path.join(base_output_dir, "dualscaffolding")
        else:
            self.output_dir = base_output_dir
        
        # API keys
        try:
            self.api_key = os.environ['OPENAI_API_KEY']
        except:
            print("[ERROR] No OpenAI key found")
        
        # Initialize clients based on model type
        if self.model_type == "gpt":
            self.client = OpenAI(api_key=self.api_key)
        elif self.model_type == "llama70b" or self.model_type == "llama8b":
            self.client = OpenAI(
                base_url="http://babel-15-36:8081/v1",
                api_key="EMPTY"
            )
        
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[INFO] Initializing conversation forecasting with{'out' if not include_intentions else ''} intentions for {dataset_type} dataset")
        print(f"[INFO] Using model: {model_type}")
        print(f"[INFO] Data source: {self.dataset_dir}")
        print(f"[INFO] Results will be saved to: {self.output_dir}")
        print(f"[INFO] Summary type: {summary_type}")
        print(f"[INFO] Conversation ratio: {ratio}")
        print(f"[INFO] Including intentions: {include_intentions}")

        self.dataset_file = self.get_dataset_file()
        if self.dataset_file:
            print(f"[INFO] Found dataset file: {os.path.basename(self.dataset_file)}")
        else:
            print(f"[ERROR] No matching dataset file found")
            exit(1)

    def get_dataset_file(self):
        ratio_str = str(self.ratio)
        file_pattern = f"ratio_{ratio_str}.csv"
        potential_file = os.path.join(self.dataset_dir, file_pattern)
        return potential_file if os.path.exists(potential_file) else None
    
    def format_conversation_with_intentions(self, dialogue_rows):
        """Format conversation from dialogue rows into a readable format with intentions."""
        formatted_conversation = []
        for _, row in dialogue_rows.iterrows():
            # Handle speaker role context for P4G dataset, as in paste-2
            speaker = row['speaker']
            if self.dataset_type == "p4g":
                if speaker == "EE":
                    speaker = "Persuadee (EE)"
                elif speaker == "ER":
                    speaker = "Persuader (ER)"
            elif self.dataset_type == "casino":
                if speaker == "mturk_agent_1":
                    speaker = "Agent 1"
                elif speaker == "mturk_agent_2":
                    speaker = "Agent 2"
            
            if self.include_intentions and 'intention' in row and pd.notna(row['intention']):
                formatted_conversation.append(f"{speaker}: {row['utterance']} [{row['intention']}]")
            else:
                formatted_conversation.append(f"{speaker}: {row['utterance']}")
            
        return "\n".join(formatted_conversation) 

    def create_prediction_prompt(self, row):
        """Create a prompt for prediction based on dialogue data or formatted conversation."""
        if 'conversation' in row:
            # This is already a formatted conversation
            conversation = row['conversation']
        else:
            # We need to format the conversation from the current row
            # This is a placeholder that will be populated in process_dataset
            conversation = "Unknown conversation"
            
        summary = ""
        if self.summary_type != "none":
            summary_column = f"{self.summary_type}_summary"
            if summary_column in row and pd.notna(row[summary_column]):
                if self.dataset_type == "cb":
                    summary = f"{row[summary_column]}"
                else:
                    summary = f"{row[summary_column]}"

        if self.dataset_type == "cb":
            buyer_target = row.get('buyer_target')
            seller_target = row.get('seller_target')

            summary_part = f", [Summary: {summary}]" if summary else ""  # Format like paste-2
            intentions_note = " with intentions" if self.include_intentions else ""
            
            prompt = f"""Analyze this negotiation, given in the format <buyer target, seller target, [negotiation{intentions_note}]{", [summary]" if summary else ""}> and predict the projected sale price that lies between the buyer and seller targets. Provide only the final answer in the format 'FINAL_PRICE: [number]'
INPUT: <${buyer_target}, ${seller_target}, [{conversation}]{summary_part}>"""

        elif self.dataset_type == "p4g":
            donation_amount = row.get('donation_amount')

            intentions_note = " with intentions" if self.include_intentions else ""
            summary_part = f" {summary}" if summary else ""  # Format like paste-2
            
            prompt = f"""You are helping analyze a persuasion conversation{intentions_note}. Predict whether the persuadee will make a donation on the spot at the end of this conversation. Provide your answer in the format 'DONATION: YES/NO'\n\nConversation:\n{conversation}{summary_part}"""
        elif self.dataset_type == "casino":
            # Extract agent preferences
            agent1_high = row.get('mturk_agent_1_high_item', '').lower() if pd.notna(row.get('mturk_agent_1_high_item', '')) else 'unknown'
            agent1_medium = row.get('mturk_agent_1_medium_item', '').lower() if pd.notna(row.get('mturk_agent_1_medium_item', '')) else 'unknown'
            agent1_low = row.get('mturk_agent_1_low_item', '').lower() if pd.notna(row.get('mturk_agent_1_low_item', '')) else 'unknown'
            
            agent2_high = row.get('mturk_agent_2_high_item', '').lower() if pd.notna(row.get('mturk_agent_2_high_item', '')) else 'unknown'
            agent2_medium = row.get('mturk_agent_2_medium_item', '').lower() if pd.notna(row.get('mturk_agent_2_medium_item', '')) else 'unknown'
            agent2_low = row.get('mturk_agent_2_low_item', '').lower() if pd.notna(row.get('mturk_agent_2_low_item', '')) else 'unknown'
            
            intentions_note = " with intentions" if self.include_intentions else ""
            summary_part = f" {summary}" if summary else ""
            
            prompt = f"""You are helping analyze a negotiation conversation where two agents are discussing the allocation of resources. Each resource has exactly three units that must be divided between the two agents. The total amount of each resource allocated to both agents must add up to three. 

Agent Preferences:
Agent 1: High priority: {agent1_high}, Medium priority: {agent1_medium}, Low priority: {agent1_low}
Agent 2: High priority: {agent2_high}, Medium priority: {agent2_medium}, Low priority: {agent2_low}

Conversation{intentions_note}:
{conversation}{summary_part}

Based on this negotiation, predict the final allocation of resources. Provide only your answer using the following format with curly braces, with no explanation:

OUTCOME: {{agent1:{{food:[number], water:[number], firewood:[number]}}, agent2:{{food:[number], water:[number], firewood:[number]}}}}

Remember: Each resource must sum to exactly 3 units across both agents."""
            
        return prompt

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    def query_model(self, prompt):
        try:
            if self.model_type == "gpt":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            elif self.model_type == "llama70b":
                try:
                    response = self.client.chat.completions.create(
                        model="meta-llama/Llama-3.1-70B-Instruct",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=200
                    )
                    time.sleep(0.5)
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"[ERROR] Llama API call failed: {str(e)}")
                    time.sleep(2)
                    raise
            elif self.model_type == "llama8b":
                try:
                    response = self.client.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=200
                    )
                    time.sleep(0.5)
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"[ERROR] Llama API call failed: {str(e)}")
                    time.sleep(2)
                    raise
        except Exception as e:
            print(f"[ERROR] Model API call failed: {str(e)}")
            time.sleep(2)
            raise

    def extract_allocation(self, response):
        """Extract final allocation from model output for Casino dataset."""
        ALLOCATION_PATTERN = re.compile(r'{agent1:{food:(\d+),\s*water:(\d+),\s*firewood:(\d+)},\s*agent2:{food:(\d+),\s*water:(\d+),\s*firewood:(\d+)}}', re.IGNORECASE)
        try:
            # Quick input validation
            if not response or not isinstance(response, str):
                return None
                
            # First try direct pattern matching on the original text
            braces_match = ALLOCATION_PATTERN.search(response)
            if braces_match:
                allocation = {'agent1': {}, 'agent2': {}}
                allocation['agent1']['food'] = int(braces_match.group(1))
                allocation['agent1']['water'] = int(braces_match.group(2))
                allocation['agent1']['firewood'] = int(braces_match.group(3))
                allocation['agent2']['food'] = int(braces_match.group(4))
                allocation['agent2']['water'] = int(braces_match.group(5))
                allocation['agent2']['firewood'] = int(braces_match.group(6))
                return allocation
                
            # Fall back to original logic if direct match fails
            processed_response = response
            if "<|end_header_id|>" in response:
                processed_response = response.split("<|end_header_id|>")[1].strip()
            elif "assistant" in response:
                processed_response = response.split("assistant")[1].strip()
            elif "OUTCOME" in response:
                parts = response.split("OUTCOME")
                if len(parts) > 2:
                    processed_response = "OUTCOME".join(parts[2:]).strip()
            
            # Try the regex on the processed text
            allocation = {'agent1': {}, 'agent2': {}}
            braces_match = ALLOCATION_PATTERN.search(processed_response)
            if braces_match:
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

    def calculate_utility_score(self, allocation, preferences):
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

    def extract_prediction(self, response):
        """Extract prediction from LLM response"""
        response = response.strip().lower()
        
        if self.dataset_type == "cb":
            # Extract predicted final price
            pattern = r'final_price:\s*\$?\s*\[?\s*([0-9,]+(?:\.\d+)?)\s*\]?'
            price_match = re.search(pattern, response)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                try:
                    final_price = float(price_str)
                    return final_price
                except ValueError:
                    print(f"[WARNING] Could not convert price to float: {price_match.group(1)}")
            
            # If no structured format, try to find any number in the response
            number_match = re.search(r"\$?(\d+(?:\.\d+)?)", response)
            if number_match:
                try:
                    return float(number_match.group(1))
                except ValueError:
                    pass
            
            print(f"[WARNING] Could not extract price prediction from response: {response}")
            return None
            
        elif self.dataset_type == "p4g":
            if "assistant" in response:
            # Extract content after the marker
                parts = response.split("assistant")
                if len(parts) > 1:
                    response = parts[1]
                
                # Look for DONATION: YES or DONATION: NO pattern
                donation_match = re.search(r'DONATION:\s*(YES|NO)', response, re.IGNORECASE)
                if donation_match:
                    donation_decision = donation_match.group(1).upper()
                    return donation_decision
            else:
                donation_match = re.search(r'DONATION:\s*(YES|NO)', response, re.IGNORECASE)
                if donation_match:
                    donation_decision = donation_match.group(1).upper()
                    return donation_decision
        elif self.dataset_type == "casino":
            # Use the extract_allocation function for casino dataset
            return self.extract_allocation(response)

        print(f"[WARNING] Could not extract prediction from response: {response}")
        return None


    def determine_actual_outcome(self, row):
        if self.dataset_type == "cb":
            return row['sale_price']
        elif self.dataset_type == "p4g":
            donation_made = row.get('donation_made')
            if donation_made is not None:
                return "YES" if donation_made==1 else "NO"
            return None
        elif self.dataset_type == "casino":
            try:
                # Extract actual allocation
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
                
                # Get preferences
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
                
                # Calculate utility scores
                agent1_utility = self.calculate_utility_score({'agent1': actual_allocation['agent1']}, preferences)
                agent2_utility = self.calculate_utility_score({'agent2': actual_allocation['agent2']}, preferences)
                
                return {
                    'allocation': actual_allocation,
                    'preferences': preferences,
                    'agent1_utility': agent1_utility,
                    'agent2_utility': agent2_utility
                }
            except Exception as e:
                print(f"[WARNING] Error determining actual outcome for casino: {e}")
                return None
        if 'label' in row and pd.notna(row['label']):
            return row['label']

        print(f"[WARNING] No label found for dialogue {row.get('dialogue_id', 'unknown')}")
        return None

    def check_correctness(self, actual, predicted):
        if actual is None or predicted is None:
            return 0
            
        if self.dataset_type == "cb":
            return None
        elif self.dataset_type == "p4g":
            return 1 if str(actual).lower() == str(predicted).lower() else 0
        elif self.dataset_type == "casino":
            # For casino, we use utility MSE as a metric
            return None
            
        return 0

    def process_batch(self, batch):
        """Process a batch of dialogues."""
        results = []

        for _, row in batch.iterrows():
            try:
                prompt = self.create_prediction_prompt(row)
                response = self.query_model(prompt)
                prediction = self.extract_prediction(response)
                actual = self.determine_actual_outcome(row)
                correct = self.check_correctness(actual, prediction)

                result = {
                    'dialogue_id': row.get('dialogue_id', ''),
                    'summary_type': self.summary_type,
                    'model_type': self.model_type,
                    'include_intentions': self.include_intentions,
                    'conversation': row['conversation'],
                    'prompt': prompt,
                    'response': response,
                }

                if self.dataset_type == "cb":
                    result.update({
                        'predicted_final_price': prediction,
                        'buyer_target': row.get('buyer_target'),
                        'seller_target': row.get('seller_target'),
                        'sale_price': row.get('sale_price')
                    })
                    
                    # Calculate price error if available
                    if prediction is not None and row.get('sale_price') is not None:
                        result['price_error'] = abs(prediction - row.get('sale_price'))
                    
                    # Calculate percentage error relative to price range
                    if prediction is not None and row.get('sale_price') is not None and row.get('buyer_target') is not None and row.get('seller_target') is not None:
                        price_range = abs(row.get('buyer_target') - row.get('seller_target'))
                        if price_range > 0:
                            result['percent_error'] = (result['price_error'] / price_range) * 100
                    
                elif self.dataset_type == "p4g":
                    result.update({
                        'actual': actual.lower() if actual else None,
                        'predicted': prediction.lower() if prediction else None,
                        'correct': correct,
                        'donation_amount': row.get('donation_amount')
                    })
                # When adding prediction results to the output
                elif self.dataset_type == "casino" and isinstance(actual, dict) and isinstance(prediction, dict):
                    try:
                        # Safely extract actual values
                        for agent in ['agent1', 'agent2']:
                            for resource in ['food', 'water', 'firewood']:
                                actual_val = actual.get('allocation', {}).get(agent, {}).get(resource)
                                pred_val = prediction.get(agent, {}).get(resource)

                                if actual_val is not None:
                                    result[f'{agent}_{resource}_actual'] = actual_val
                                if pred_val is not None:
                                    result[f'{agent}_{resource}_pred'] = pred_val

                        # Safely get preferences
                        preferences = actual.get('preferences', {})

                        # Compute predicted utilities only if all values present
                        if all(agent in prediction for agent in ['agent1', 'agent2']):
                            pred_agent1_utility = self.calculate_utility_score({'agent1': prediction['agent1']}, preferences)
                            pred_agent2_utility = self.calculate_utility_score({'agent2': prediction['agent2']}, preferences)

                            result['agent1_utility_pred'] = pred_agent1_utility
                            result['agent2_utility_pred'] = pred_agent2_utility

                            if 'agent1_utility' in actual:
                                result['agent1_utility_actual'] = actual['agent1_utility']
                                result['utility_mse'] = (actual['agent1_utility'] - pred_agent1_utility) ** 2

                            if 'agent2_utility' in actual:
                                result['agent2_utility_actual'] = actual['agent2_utility']


                    except Exception as e:
                        print(f"[WARNING] Failed to safely extract actual/predicted for casino: {e}")

                results.append(result)
                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Failed to process dialogue {row.get('dialogue_id', 'unknown')}: {str(e)}")

        return results

    def process_dataset(self):
        print(f"[INFO] Processing file: {self.dataset_file}")
        df = pd.read_csv(self.dataset_file)
        print(f"[INFO] Loaded {len(df)} rows from dataset")

        # For utterance-level data, group by dialogue_id to create complete conversations
        if 'utterance_idx' in df.columns and 'speaker' in df.columns and 'utterance' in df.columns:
            print(f"[INFO] Found utterance-level data, grouping by dialogue_id")
            dialogue_groups = df.groupby('dialogue_id')
            print(f"[INFO] Found {len(dialogue_groups)} unique dialogues")
            
            # Create a dataframe with one row per dialogue
            dialogues_df = []
            for dialogue_id, group in dialogue_groups:
                # Sort by utterance index
                group = group.sort_values('utterance_idx')
                
                # Format the conversation with intentions if required
                conversation = self.format_conversation_with_intentions(group)
                
                # Create a row for this dialogue with metadata and conversation
                dialogue_row = {
                    'dialogue_id': dialogue_id,
                    'conversation': conversation,
                }
                
                # Add dataset-specific fields
                if self.dataset_type == "cb":
                    dialogue_row.update({
                        'buyer_target': group['buyer_target'].iloc[0] if 'buyer_target' in group.columns else None,
                        'seller_target': group['seller_target'].iloc[0] if 'seller_target' in group.columns else None,
                        'sale_price': group['sale_price'].iloc[0] if 'sale_price' in group.columns else None
                    })
                elif self.dataset_type == "p4g":
                    # For the p4g dataset, correctly handle the donation_made column
                    dialogue_row.update({
                        'donation_amount': group['donation_amount'].iloc[0] if 'donation_amount' in group.columns else None,
                        'donation_made': group['donation_made'].iloc[0] if 'donation_made' in group.columns else None
                    })
                elif self.dataset_type == "casino":
                    # Add casino-specific fields
                    casino_fields = [
                        'mturk_agent_1_high_item', 'mturk_agent_1_medium_item', 'mturk_agent_1_low_item',
                        'mturk_agent_2_high_item', 'mturk_agent_2_medium_item', 'mturk_agent_2_low_item',
                        'mturk_agent_1_food', 'mturk_agent_1_water', 'mturk_agent_1_firewood',
                        'mturk_agent_2_food', 'mturk_agent_2_water', 'mturk_agent_2_firewood'
                    ]
                    
                    for field in casino_fields:
                        if field in group.columns:
                            dialogue_row[field] = group[field].iloc[0]
                
                if 'label' in group.columns:
                    dialogue_row['label'] = group['label'].iloc[0]
                
                # Add summary if available
                if self.summary_type != "none":
                    summary_column = f"{self.summary_type}_summary"
                    if summary_column in group.columns:
                        dialogue_row[summary_column] = group[summary_column].iloc[0]
                
                dialogues_df.append(dialogue_row)
            
            # Convert list of dictionaries to DataFrame
            dialogues_df = pd.DataFrame(dialogues_df)
            print(f"[INFO] Created {len(dialogues_df)} dialogue entries")
            df = dialogues_df
        else:
            print(f"[INFO] Using provided conversation data (not utterance-level)")
            
        all_results = []
        for i in tqdm(range(0, len(df), self.batch_size), desc="Processing", unit="batch"):
            batch = df.iloc[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
            self.save_progress(all_results)

        return all_results

    def save_progress(self, results):
        if not results:
            return

        df = pd.DataFrame(results)
        ratio_str = str(self.ratio)
        intentions_str = "with_intentions" if self.include_intentions else "no_intentions"
        self.progress_output_path = os.path.join(
            self.output_dir, 
            f"{self.dataset_type}_{ratio_str}_{self.summary_type}_{intentions_str}_{self.model_type}_results.csv"
        )
        df.to_csv(self.progress_output_path, index=False)
        print(f"[INFO] Progress saved: {len(results)} dialogues processed to {self.progress_output_path}")

    def run(self):
        results = self.process_dataset()
        if results:
            df = pd.DataFrame(results)
            print("\n[INFO] Processing complete.")
            print(f"  Total samples processed: {len(df)}")
            
            if self.dataset_type == "cb":
                # Calculate price prediction metrics
                if 'price_error' in df.columns:
                    mean_error = df['price_error'].mean()
                    median_error = df['price_error'].median()
                    print(f"  Mean absolute price prediction error: ${mean_error:.2f}")
                    print(f"  Median absolute price prediction error: ${median_error:.2f}")
                    
                    # Calculate percentage error relative to price range
                    if 'percent_error' in df.columns:
                        mean_percent_error = df['percent_error'].mean()
                        print(f"  Mean percentage error (relative to price range): {mean_percent_error:.2f}%")
            
                    # Calculate if predictions are within price range
                    df['within_range'] = df.apply(
                        lambda row: 1 if (row['buyer_target'] <= row['predicted_final_price'] <= row['seller_target']) or
                                        (row['seller_target'] <= row['predicted_final_price'] <= row['buyer_target']) 
                                   else 0, axis=1)
                    print(f"  Predictions within negotiation range: {df['within_range'].sum()} ({df['within_range'].mean() * 100:.2f}%)")
            elif self.dataset_type == "p4g":
                # For p4g dataset with binary prediction
                if 'correct' in df.columns:
                    print(f"  Correct predictions: {df['correct'].sum()} ({df['correct'].mean() * 100:.2f}%)")
                    
                    # Calculate precision, recall, F1 if applicable
                    if 'actual' in df.columns and 'predicted' in df.columns:
                        # Filter out None values
                        valid_predictions = df.dropna(subset=['actual', 'predicted'])
                        if len(valid_predictions) > 0:
                            # Convert yes/no to 1/0
                            y_true = [1 if pred.lower() == 'yes' else 0 for pred in valid_predictions['actual']]
                            y_pred = [1 if pred.lower() == 'yes' else 0 for pred in valid_predictions['predicted']]
                            
                            precision = precision_score(y_true, y_pred, zero_division=0)
                            recall = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1 Score: {f1:.4f}")
            elif self.dataset_type == "casino":
                # Calculate utility MSE for casino dataset
                if 'utility_mse' in df.columns:
                    df_valid = df.dropna(subset=['utility_mse'])
                    if len(df_valid) > 0:
                        mean_utility_mse = df_valid['utility_mse'].mean()
                        median_utility_mse = df_valid['utility_mse'].median()
                        print(f"  Mean Utility MSE: {mean_utility_mse:.4f}")
                        print(f"  Median Utility MSE: {median_utility_mse:.4f}")
                        
                        # Calculate exact match ratio for resource allocation
                        if all(col in df.columns for col in ['agent1_food_actual', 'agent1_food_pred']):
                            df['food_match'] = df.apply(
                                lambda row: 1 if row['agent1_food_actual'] == row['agent1_food_pred'] and 
                                              row['agent2_food_actual'] == row['agent2_food_pred'] else 0, axis=1)
                            df['water_match'] = df.apply(
                                lambda row: 1 if row['agent1_water_actual'] == row['agent1_water_pred'] and 
                                              row['agent2_water_actual'] == row['agent2_water_pred'] else 0, axis=1)
                            df['firewood_match'] = df.apply(
                                lambda row: 1 if row['agent1_firewood_actual'] == row['agent1_firewood_pred'] and 
                                              row['agent2_firewood_actual'] == row['agent2_firewood_pred'] else 0, axis=1)
                            
                            print(f"  Food allocation accuracy: {df['food_match'].mean() * 100:.2f}%")
                            print(f"  Water allocation accuracy: {df['water_match'].mean() * 100:.2f}%")
                            print(f"  Firewood allocation accuracy: {df['firewood_match'].mean() * 100:.2f}%")
                            
                            # Calculate exact resource allocation match for all resources
                            df['exact_match'] = df.apply(
                                lambda row: 1 if row['food_match'] == 1 and 
                                              row['water_match'] == 1 and 
                                              row['firewood_match'] == 1 else 0, axis=1)
                            print(f"  Exact allocation match: {df['exact_match'].sum()} ({df['exact_match'].mean() * 100:.2f}%)")
                
                
            # Save final results with metrics
            ratio_str = str(self.ratio)
            intentions_str = "with_intentions" if self.include_intentions else "no_intentions"
            final_output_path = os.path.join(
                self.output_dir, 
                f"{self.dataset_type}_{ratio_str}_{self.summary_type}_{intentions_str}_{self.model_type}_final_results.csv"
            )
            df.to_csv(final_output_path, index=False)
            
            # Delete partial results file if it exists
            if hasattr(self, 'progress_output_path') and os.path.exists(self.progress_output_path):
                try:
                    os.remove(self.progress_output_path)
                    print(f"[INFO] Deleted intermediate results file: {self.progress_output_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to delete intermediate results file: {str(e)}")
            
            print(f"[INFO] Final results saved to: {final_output_path}")
        else:
            print("[ERROR] No results generated during processing.")

def main():
    parser = argparse.ArgumentParser(description="Conversation Forecasting With Intentions")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["cb", "p4g", "casino"],
                        help="Type of dataset (cb for Craigslist Bargain, p4g for Persuasion for Good)")
    parser.add_argument("--model_type", type=str, default="gpt", choices=["gpt", "llama8b", "llama70b"],
                        help="Type of model to use (gpt for OpenAI, llama for open source Llama model)")
    parser.add_argument("--summary_type", type=str, default="none",
                        choices=["none", "traditional", "scd", "relational", "scm", "appraisal_theory", "politeness_theory_stage2"],
                        help="Type of summary to use ('none' for no summary)")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Ratio of conversation used (e.g., 0.5 for 50%)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of dialogues to process in each batch")
    parser.add_argument("--include_intentions", action="store_true",
                        help="Include speaker intentions in the conversation format")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed")


    args = parser.parse_args()

    seed_everything(args.seed)

    forecaster = ConversationForecastingWithIntentions(
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        summary_type=args.summary_type,
        ratio=args.ratio,
        batch_size=args.batch_size,
        include_intentions=args.include_intentions
    )

    forecaster.run()

if __name__ == "__main__":
    main()


# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type none --ratio 0.5 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type none --ratio 0.625 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type none --ratio 0.75 --include_intentions

# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.375 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.5 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.625 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.75 

# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.25 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.375 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.5 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.625 
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.75 

# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.25 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.375 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.5 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.625 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scm --ratio 0.75 --include_intentions

# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.25 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.375 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.5 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.625 --include_intentions
# python zero_shot.py --dataset_type p4g --model_type llama70b --summary_type scd --ratio 0.75 --include_intentions