import pandas as pd
import os
import math
import argparse

def main(input_path, output_dir, ratios):
    # Load dataset
    df = pd.read_csv(input_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if 'worker_score_bucket' in df.columns:
        df = df.drop(columns=['worker_score_bucket'])
        
    # Identify dialogue ID and utterance index columns
    if 'dialogue_id' not in df.columns or 'utterance_idx' not in df.columns:
        raise ValueError("Input CSV must contain 'dialogue_id' and 'utterance_idx' columns.")

    for ratio in ratios:
        output_rows = []

        # Group by dialogue_id
        grouped = df.groupby('dialogue_id')

        for dialogue_id, group in grouped:
            group = group.sort_values(by='utterance_idx')
            keep_n = max(1, math.floor(len(group) * ratio))
            truncated = group.head(keep_n)
            output_rows.append(truncated)

        # Combine and export
        result_df = pd.concat(output_rows)
        output_file = os.path.join(output_dir, f'ratio_{ratio}.csv')
        result_df.to_csv(output_file, index=False)

    print(f"Files saved in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate dialogue utterances based on ratios.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for truncated CSVs.")
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.25, 0.375, 0.5, 0.625, 0.75],
                        help="List of ratios to sample from each dialogue.")

    args = parser.parse_args()
    main(args.input, args.output, args.ratios)
