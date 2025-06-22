import pandas as pd
import os
import argparse
import openai
import json
import time
from typing import List, Dict, Tuple

""" Segment conversations into "beginning", "middle", and "end" segments using GPT-4o """

def setup_openai_client(api_key: str):
    """Initialize OpenAI client with API key."""
    return openai.OpenAI(api_key=api_key)

def format_conversation_for_gpt(group: pd.DataFrame) -> str:
    """Format conversation utterances for GPT analysis."""
    conversation = []
    for _, row in group.iterrows():
        # Adjust these column names based on your actual CSV structure
        # Common column names: 'text', 'message', 'utterance', 'content'
        utterance_text = row.get('text', row.get('message', row.get('utterance', row.get('content', str(row.iloc[-1])))))
        speaker = row.get('speaker', row.get('role', f"Speaker_{row['utterance_idx']}"))
        conversation.append(f"{speaker}: {utterance_text}")
    
    return "\n".join(conversation)

def get_segmentation_from_gpt(client, conversation_text: str, dialogue_id: str) -> Dict[str, int]:
    """Get conversation segmentation from GPT-4o."""
    
    prompt = f"""Analyze the following conversation and divide it into three semantic segments: beginning, middle, and end.

The conversation has utterances numbered from 0 to N. Please identify:
1. Where the "beginning" section ends (last utterance index of beginning)
2. Where the "middle" section ends (last utterance index of middle)

The "end" section will automatically be from middle+1 to the final utterance.

Conversation:
{conversation_text}

Please respond with ONLY a JSON object in this exact format:
{{"beginning_end": <utterance_index>, "middle_end": <utterance_index>}}

Where utterance_index corresponds to the utterance_idx values in the conversation."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a conversation analyst. Analyze conversations and segment them into beginning, middle, and end sections based on semantic content and conversation flow."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response (handle markdown code blocks)
        try:
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                # Find the actual JSON content between code blocks
                lines = response_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                response_text = '\n'.join(json_lines).strip()
            
            segmentation = json.loads(response_text)
            
            # Validate the response has required keys
            if 'beginning_end' not in segmentation or 'middle_end' not in segmentation:
                print(f"Warning: Invalid segmentation format for dialogue {dialogue_id}. Missing required keys.")
                return None
                
            return segmentation
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON for dialogue {dialogue_id}. Response: {response_text}")
            print(f"JSON Error: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error processing dialogue {dialogue_id}: {str(e)}")
        return None

def segment_conversation(group: pd.DataFrame, segmentation: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Segment conversation based on GPT's analysis."""
    group = group.sort_values(by='utterance_idx')
    
    beginning_end_idx = segmentation['beginning_end']
    middle_end_idx = segmentation['middle_end']
    
    # Beginning: from start to beginning_end_idx (inclusive)
    beginning = group[group['utterance_idx'] <= beginning_end_idx]
    
    # Middle: from start to middle_end_idx (inclusive) - this includes beginning + middle
    middle = group[group['utterance_idx'] <= middle_end_idx]
    
    # End: all utterances (beginning + middle + end)
    end = group.copy()
    
    return beginning, middle, end

def main(input_path: str, output_dir: str, api_key: str, delay: float):
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove worker_score_bucket if present
    if 'worker_score_bucket' in df.columns:
        df = df.drop(columns=['worker_score_bucket'])
    
    # Verify required columns
    if 'dialogue_id' not in df.columns or 'utterance_idx' not in df.columns:
        raise ValueError("Input CSV must contain 'dialogue_id' and 'utterance_idx' columns.")
    
    # Initialize OpenAI client
    client = setup_openai_client(api_key)
    
    # Initialize output DataFrames
    beginning_rows = []
    middle_rows = []
    end_rows = []
    
    # Group by dialogue_id
    grouped = df.groupby('dialogue_id')
    total_dialogues = len(grouped)
    
    print(f"Processing {total_dialogues} dialogues...")
    
    for i, (dialogue_id, group) in enumerate(grouped, 1):
        print(f"Processing dialogue {i}/{total_dialogues}: {dialogue_id}")
        
        # Format conversation for GPT
        conversation_text = format_conversation_for_gpt(group)
        
        # Get segmentation from GPT
        segmentation = get_segmentation_from_gpt(client, conversation_text, dialogue_id)
        
        if segmentation is None:
            print(f"Skipping dialogue {dialogue_id} due to segmentation error.")
            continue
        
        try:
            # Segment the conversation
            beginning, middle, end = segment_conversation(group, segmentation)
            
            beginning_rows.append(beginning)
            middle_rows.append(middle)
            end_rows.append(end)
            
        except Exception as e:
            print(f"Error segmenting dialogue {dialogue_id}: {str(e)}")
            continue
        
        # Add delay to respect API rate limits
        if delay > 0:
            time.sleep(delay)
    
    # Combine and export results
    if beginning_rows:
        beginning_df = pd.concat(beginning_rows, ignore_index=True)
        beginning_file = os.path.join(output_dir, 'beginning.csv')
        beginning_df.to_csv(beginning_file, index=False)
        print(f"Beginning segments saved to: {beginning_file}")
    
    if middle_rows:
        middle_df = pd.concat(middle_rows, ignore_index=True)
        middle_file = os.path.join(output_dir, 'middle.csv')
        middle_df.to_csv(middle_file, index=False)
        print(f"Middle segments (beginning + middle) saved to: {middle_file}")
    
    if end_rows:
        end_df = pd.concat(end_rows, ignore_index=True)
        end_file = os.path.join(output_dir, 'end.csv')
        end_df.to_csv(end_file, index=False)
        print(f"End segments (full conversations) saved to: {end_file}")
    
    print(f"All files saved in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment dialogue utterances using GPT-4o analysis.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for segmented CSVs.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds (default: 1.0).")
    
    args = parser.parse_args()
    api_key = os.environ["OPENAI_API_KEY"]
    main(args.input, args.output, api_key, args.delay)