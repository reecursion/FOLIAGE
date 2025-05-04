#!/usr/bin/env python3
"""
Simple script to clean casino datasets by removing dialogues with NaN values.
"""

import os
import pandas as pd
from pathlib import Path
import argparse

def clean_casino_datasets(directory_path, output_dir=None):
    """
    Process all CSV files in a directory and remove dialogues with NaN values
    in casino-related columns.
    
    Args:
        directory_path: Path to directory containing CSV files
        output_dir: Directory to save cleaned files (defaults to same dir with '_cleaned' suffix)
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Find all CSV files in the directory
    csv_files = list(Path(directory_path).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Casino-related columns that should not have NaN values
    casino_columns = [
        'mturk_agent_1_high_item', 'mturk_agent_1_medium_item', 'mturk_agent_1_low_item',
        'mturk_agent_2_high_item', 'mturk_agent_2_medium_item', 'mturk_agent_2_low_item',
        'mturk_agent_1_food', 'mturk_agent_1_water', 'mturk_agent_1_firewood',
        'mturk_agent_2_food', 'mturk_agent_2_water', 'mturk_agent_2_firewood'
    ]
    
    # Process each file
    total_files = 0
    total_dialogues_original = 0
    total_dialogues_removed = 0
    
    for file_path in csv_files:
        print(f"\nProcessing: {file_path}")
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
            total_files += 1
            
            # Get original dialogue count
            original_dialogues = df['dialogue_id'].unique()
            original_count = len(original_dialogues)
            total_dialogues_original += original_count
            print(f"  Original file has {len(df)} rows with {original_count} unique dialogues")
            
            # Check which columns exist
            existing_columns = [col for col in casino_columns if col in df.columns]
            if not existing_columns:
                print("  No casino-related columns found, skipping file")
                continue
            
            # Find dialogues with NaN values
            dialogues_with_nan = []
            for dialogue_id, group in df.groupby('dialogue_id'):
                if group[existing_columns].isna().any().any():
                    dialogues_with_nan.append(dialogue_id)
            
            # Skip if no NaN values found
            if not dialogues_with_nan:
                print("  No dialogues with NaN values found, skipping file")
                
                # Save to output directory if specified
                if output_dir:
                    output_path = Path(output_dir) / Path(file_path).name
                    df.to_csv(output_path, index=False)
                    print(f"  Copied file to {output_path}")
                continue
            
            # Remove dialogues with NaN values
            print(f"  Found {len(dialogues_with_nan)} dialogues with NaN values")
            df_clean = df[~df['dialogue_id'].isin(dialogues_with_nan)]
            cleaned_count = len(df_clean['dialogue_id'].unique())
            removed_count = original_count - cleaned_count
            total_dialogues_removed += removed_count
            
            # Save cleaned file
            if output_dir:
                output_path = Path(output_dir) / f"{Path(file_path).stem}_cleaned.csv"
            else:
                output_path = Path(file_path).with_name(f"{Path(file_path).stem}_cleaned.csv")
            
            df_clean.to_csv(output_path, index=False)
            print(f"  Saved cleaned file to {output_path}")
            print(f"  Original: {original_count} dialogues, Cleaned: {cleaned_count} dialogues, Removed: {removed_count} dialogues")
            
            # Print removed dialogues
            if len(dialogues_with_nan) <= 20:  # Limit output if too many
                print("  Removed dialogue IDs:")
                for dialogue_id in dialogues_with_nan:
                    print(f"    - {dialogue_id}")
            else:
                print(f"  Removed {len(dialogues_with_nan)} dialogue IDs (too many to display)")
                
        except Exception as e:
            print(f"  Error processing file: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Processed {total_files} files")
    print(f"Total original dialogues: {total_dialogues_original}")
    print(f"Total dialogues removed: {total_dialogues_removed}")
    print(f"Total dialogues remaining: {total_dialogues_original - total_dialogues_removed}")
    print(f"Removal percentage: {(total_dialogues_removed / total_dialogues_original * 100):.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Clean casino datasets by removing dialogues with NaN values")
    parser.add_argument("--directory", help="Directory containing casino dataset CSV files")
    parser.add_argument("--output-dir", "-o", help="Directory to save cleaned files (default: same as input with '_cleaned' suffix)")
    
    args = parser.parse_args()
    clean_casino_datasets(args.directory, args.output_dir)

if __name__ == "__main__":
    main()