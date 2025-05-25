import os
import pandas as pd

def print_nan_utility_mse_rows(root_dir):
    total_nan_rows = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path)
                    if "utility_mse" in df.columns:
                        nan_rows = df[df["utility_mse"].isna()]
                        if not nan_rows.empty:
                            print(f"\n{file_path} — {len(nan_rows)} NaN rows in 'utility_mse':")
                            print(nan_rows)
                            total_nan_rows += len(nan_rows)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    print(f"\nTotal NaN rows in 'utility_mse' column across all files: {total_nan_rows}")
    return total_nan_rows

# Example usage:
root_directory = "src/sft/results/casino"
print_nan_utility_mse_rows(root_directory)
