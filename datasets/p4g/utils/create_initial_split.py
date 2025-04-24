import pandas as pd
import os
import math

# Load dataset
df = pd.read_csv('/home/gganeshl/FOLIAGE/datasets/p4g/utterance_intentions.csv')

# Ensure the output directory exists
os.makedirs('/home/gganeshl/FOLIAGE/datasets/p4g/final', exist_ok=True)

# Desired ratios
ratios = [0.25, 0.375, 0.5, 0.625, 0.75]

# Desired column order and rename mapping
column_order = ['dialogue_id', 'donation_amount', 'donation_made', 
                'utterance_idx', 'speaker', 'utterance', 'intention']
rename_map = {'gpt-4o_intention': 'intention'}

# Process for each ratio
for ratio in ratios:
    output_rows = []

    # Group by dialogue_id
    grouped = df.groupby('dialogue_id')

    for dialogue_id, group in grouped:
        # Sort by utterance_idx just in case
        group = group.sort_values(by='utterance_idx')
        
        # Number of rows to keep
        keep_n = max(1, math.floor(len(group) * ratio))

        # Keep top `keep_n` utterances
        truncated = group.head(keep_n)

        output_rows.append(truncated)

    # Combine and format
    result_df = pd.concat(output_rows)
    result_df = result_df.rename(columns=rename_map)
    result_df = result_df[column_order]

    # Save to CSV
    filename = f'/home/gganeshl/FOLIAGE/datasets/p4g/final/ratio_{ratio}.csv'
    result_df.to_csv(filename, index=False)

print("Files saved in 'final/' directory.")
