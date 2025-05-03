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
        
        # Base output directory
        base_output_dir = f"/home/gganeshl/FOLIAGE/src/icl/results/{dataset_type}/{model_type}"
        
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
        self.api_key = s.environ['OPENAI_API_KEY']
        
        # Initialize clients based on model type
        if self.model_type == "gpt":
            self.client = OpenAI(api_key=self.api_key)
        elif self.model_type == "llama70b" or self.model_type == "llama8b":
            self.client = OpenAI(
                base_url="http://babel-4-37:8081/v1",
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
                    speaker = "Persuadee"
                elif speaker == "ER":
                    speaker = "Persuader"
            
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
        return prompt

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    def query_model(self, prompt):
        try:
            if self.model_type == "gpt":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
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
            donation_match = re.search(r'DONATION:\s*(YES|NO)', text, re.IGNORECASE)
            if donation_match:
                donation_decision = donation_match.group(1).upper()
                return donation_decision

        print(f"[WARNING] Could not extract prediction from response: {response}")
        return None

    def determine_actual_outcome(self, row):
        if self.dataset_type == "cb":
            return row['sale_price']
        elif self.dataset_type == "p4g":
            donation_made = row.get('donation_made', None)
            if donation_made is not None:
                return "YES" if donation_made==1 else "NO"
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
                    'buyer_target': group['buyer_target'].iloc[0] if 'buyer_target' in group.columns else None,
                    'seller_target': group['seller_target'].iloc[0] if 'seller_target' in group.columns else None,
                    'sale_price': group['sale_price'].iloc[0] if 'sale_price' in group.columns else None
                }
                
                # Add other relevant data if available
                if 'donation_amount' in group.columns:
                    dialogue_row['donation_amount'] = group['donation_amount'].iloc[0]
                
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
            else:
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
    parser.add_argument("--dataset_type", type=str, required=True, choices=["cb", "p4g"],
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

    args = parser.parse_args()

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