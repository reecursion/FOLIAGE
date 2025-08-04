import os
import pandas as pd
from collections import defaultdict

# Set input and output directories
input_dir = "/home/gganeshl/FOLIAGE/datasets/casino/final - unfiltered"
output_dir = "/home/gganeshl/FOLIAGE/datasets/casino/final"
os.makedirs(output_dir, exist_ok=True)

# Function to find valid dialogue IDs
def is_valid_dialogue(df):
    totals = df.groupby("dialogue_id").agg({
        "mturk_agent_1_food": "first",
        "mturk_agent_1_water": "first",
        "mturk_agent_1_firewood": "first",
        "mturk_agent_2_food": "first",
        "mturk_agent_2_water": "first",
        "mturk_agent_2_firewood": "first",
    })

    def condition(row):
        food_total = row["mturk_agent_1_food"] + row["mturk_agent_2_food"]
        water_total = row["mturk_agent_1_water"] + row["mturk_agent_2_water"]
        firewood_total = row["mturk_agent_1_firewood"] + row["mturk_agent_2_firewood"]
        return food_total == 3 and water_total == 3 and firewood_total == 3

    return totals[totals.apply(condition, axis=1)].index.tolist()

# For tracking
original_ids = defaultdict(int)
filtered_ids = defaultdict(int)

# Loop through files
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        original_ids[filename] = df["dialogue_id"].nunique()

        valid_dialogues = is_valid_dialogue(df)
        filtered_df = df[df["dialogue_id"].isin(valid_dialogues)]

        filtered_ids[filename] = filtered_df["dialogue_id"].nunique()

        # Save
        output_path = os.path.join(output_dir, filename)
        filtered_df.to_csv(output_path, index=False)

# Print stats
print("Filtering Summary:\n")
for fname in original_ids:
    print(f"{fname}:")
    print(f"  Original dialogue count: {original_ids[fname]}")
    print(f"  Retained dialogue count: {filtered_ids[fname]}")
    print()
