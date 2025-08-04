import pandas as pd
import json

# Load dialogue data
df_dialogue = pd.read_csv("/home/gganeshl/FOLIAGE/datasets/jobinterview/utils/utterance_intentions.csv")

# Load outcomes from JSON
with open("/home/gganeshl/projecting-conversational-outcomes/Datasets/Job_Interview/data.json", "r") as f:
    data = json.load(f)

# Extract accepted solutions
dialogue_outcomes = {}
for entry in data:
    dialogue_id = entry["id"]
    for solution in entry.get("solutions", []):
        if solution.get("accepted"):
            solution_details = solution["body"]
            solution_details["dialogue_id"] = dialogue_id
            dialogue_outcomes[dialogue_id] = solution_details
            break 

# Create a DataFrame from accepted outcomes
df_outcomes = pd.DataFrame(dialogue_outcomes.values())

# Merge with dialogue data
df_final = df_dialogue.merge(df_outcomes, on="dialogue_id", how="inner")

# Save the merged data
df_final.to_csv("/home/gganeshl/FOLIAGE/datasets/jobinterview/final_dialogues_with_outcomes.csv", index=False)

print("Saved to final_dialogues_with_outcomes.csv")
