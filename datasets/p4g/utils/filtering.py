import pandas as pd

# Load the dataset
df = pd.read_csv("/home/gganeshl/FOLIAGE/datasets/p4g/utils/utterance_intentions.csv")

# Convert donation_amount to numeric (if not already), handle any errors
df['donation_amount'] = pd.to_numeric(df['donation_amount'], errors='coerce')

# Filter: keep only rows with donation_amount == 0 or > 0.1
filtered_df = df[(df['donation_amount'] == 0) | (df['donation_amount'] > 0.1)].copy()

# Set donation_made: 1 if amount > 0.1, else 0
filtered_df['donation_made'] = filtered_df['donation_amount'].apply(lambda x: 1 if x > 0.1 else 0)

# Save to p.csv
filtered_df.to_csv("/home/gganeshl/FOLIAGE/datasets/p4g/utterance_intentions.csv", index=False)

print("Saved filtered dataset to utterance_intentions.csv.")
