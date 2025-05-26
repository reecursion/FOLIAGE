import pandas as pd
import os

# Define paths to the two CSV files
csv1_path = 'src/sft/results/array/craigslistbargain/seed_10623/fold4_cb_ratio_0.5_both_traditional_predictions.csv'
csv2_path = 'src/sft/results/cb/seed_10623/cb_ratio_0.5_both_traditional_predictions.csv'

# Load the CSV files
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Define the output path in the second directory
output_path = os.path.join(os.path.dirname(csv2_path), 'cb_ratio_0.5_both_traditional_predictions.csv')

# Save the combined DataFrame
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to: {output_path}")
