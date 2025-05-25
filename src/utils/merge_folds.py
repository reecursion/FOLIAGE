import os
import re
import pandas as pd
from collections import defaultdict

def combine_folds_by_suffix(directory):
    grouped_files = defaultdict(list)

    # Match files like fold1_suffix.csv and extract "_suffix.csv"
    pattern = re.compile(r"^fold\d+(_.+\.csv)$")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            suffix = match.group(1)  # e.g., _casino_ratio_0.5_both_scd_predictions.csv
            grouped_files[suffix].append(filename)

    # Combine files for each suffix group
    for suffix, files in grouped_files.items():
        dfs = [pd.read_csv(os.path.join(directory, f)) for f in sorted(files)]
        combined_df = pd.concat(dfs, ignore_index=True)

        # Remove the leading underscore from the suffix for the output filename
        output_filename = suffix[1:]
        output_path = os.path.join("src/sft/results/casino/seed_10623", output_filename)

        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined file: {output_path}")

# Example usage
combine_folds_by_suffix("src/sft/results/array/casino/seed_10623")
