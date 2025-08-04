import os
import pandas as pd

# Paths to the two directories
dir1 = 'src/sft/results/craigslistbargain/seed_42'
dir2 = 'src/sft/results/craigslistbargain/seed_11611'

# Get list of CSV files in both directories
csvs1 = [f for f in os.listdir(dir1) if f.endswith('.csv')]
csvs2 = [f for f in os.listdir(dir2) if f.endswith('.csv')]

# Find matching file names
common_files = set(csvs1).intersection(csvs2)

# Track identical files
identical_files = []

for filename in common_files:
    path1 = os.path.join(dir1, filename)
    path2 = os.path.join(dir2, filename)
    try:
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        if df1.equals(df2):
            identical_files.append(filename)
    except Exception as e:
        print(f"Error comparing {filename}: {e}")

# Output
if identical_files:
    print("Identical CSV files in both directories:")
    for f in identical_files:
        print(f" - {f}")
else:
    print("No identical CSV files found.")
