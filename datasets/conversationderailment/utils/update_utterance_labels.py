import pandas as pd

# Replace this with your actual CSV file path
csv_path = "datasets/conversationderailment/utterance_intentions.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Ensure utterance_idx is numeric (in case it's read as string)
df['utterance_idx'] = pd.to_numeric(df['utterance_idx'], errors='coerce')

# Get the last utterance's personal_attack for each dialogue_id
last_attack_map = (
    df.sort_values(['dialogue_id', 'utterance_idx'])
      .groupby('dialogue_id')
      .tail(1)
      .set_index('dialogue_id')['personal_attack']
)

# Map the last utterance's personal_attack to all rows in the same dialogue_id
df['personal_attack'] = df['dialogue_id'].map(last_attack_map)

# Save the updated DataFrame back to the same file
df.to_csv(csv_path, index=False)