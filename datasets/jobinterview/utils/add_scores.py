import pandas as pd
import json
from typing import Dict, List, TypeVar

Option = TypeVar('Option', str, int)

def calc_score(user_role: str, user_utilities: List[Dict], bid_options: Dict):
    """Calculate a score that the user can earn by the bid."""
    
    score = 0
    for issue_name, option in bid_options.items():
        issue_utility = None
        for u in user_utilities:
            if u["name"] == issue_name:
                issue_utility = u
        
        if not issue_utility:
            continue
            
        issue_weight = issue_utility["weight"]

        if issue_utility["type"] == "INTEGER":
            option_max = issue_utility["max"]
            option_min = issue_utility["min"]
            option_value = int(option)  # Convert to int if it's a string
            if user_role == "recruiter":
                score += issue_weight * (option_max - option_value) / (option_max - option_min)
            elif user_role == "worker":
                score += issue_weight * (option_value - option_min) / (option_max - option_min)
            else:
                raise Exception(f"No such role: {user_role}")
        elif issue_utility["type"] == "DISCRETE":
            if "relatedTo" in issue_utility:
                option_weight = None
                related_issue_name = issue_utility["relatedTo"]
                for o in issue_utility["options"]:
                    if (o["names"][related_issue_name] == bid_options.get(related_issue_name) and 
                        o["names"][issue_name] == option):
                        option_weight = o["weight"]
                if option_weight is not None:
                    score += option_weight * issue_weight
            else:
                option_weight = None
                for o in issue_utility["options"]:
                    if o["name"] == option:
                        option_weight = o["weight"]
                if option_weight is not None:
                    score += option_weight * issue_weight
    
    return score

def extract_utilities(users):
    """Extract utilities as flat columns"""
    utilities_dict = {}
    
    for user in users:
        role = user["role"]
        
        # Extract weights for each utility
        for utility in user["utilities"]:
            name = utility["name"]
            weight = utility["weight"]
            utilities_dict[f"{role}_{name}_weight"] = weight
            
            # For discrete options, extract individual option weights
            if utility["type"] == "DISCRETE":
                if "relatedTo" not in utility:
                    for option in utility["options"]:
                        option_name = option["name"]
                        option_weight = option["weight"]
                        utilities_dict[f"{role}_{name}_{option_name}_weight"] = option_weight
                else:
                    # Handle related options (like Position related to Company)
                    for option in utility["options"]:
                        names = option["names"]
                        option_weight = option["weight"]
                        option_key = "_".join([f"{k}_{v}" for k, v in names.items()])
                        utilities_dict[f"{role}_{option_key}_weight"] = option_weight
    
    return utilities_dict

# Load dialogue data
df_dialogue = pd.read_csv("/home/gganeshl/FOLIAGE/datasets/jobinterview/utils/utterance_intentions.csv")

# Load outcomes from JSON
with open("/home/gganeshl/projecting-conversational-outcomes/Datasets/Job_Interview/data.json", "r") as f:
    data = json.load(f)

# Extract accepted solutions and calculate scores
dialogue_outcomes = {}
for entry in data:
    dialogue_id = entry["id"]
    users = entry["users"]
    
    for solution in entry.get("solutions", []):
        if solution.get("accepted"):
            solution_details = solution["body"].copy()
            solution_details["dialogue_id"] = dialogue_id
            
            # Calculate and add scores for each user
            for user in users:
                user_id = user["id"]
                user_role = user["role"]
                user_utilities = user["utilities"]
                user_score = calc_score(user_role, user_utilities, solution["body"])
                solution_details[f"score_{user_role}"] = user_score
            
            # Add utility weights as columns
            utilities_dict = extract_utilities(users)
            solution_details.update(utilities_dict)
            
            dialogue_outcomes[dialogue_id] = solution_details
            break

# Create a DataFrame from accepted outcomes
df_outcomes = pd.DataFrame(dialogue_outcomes.values())

# Merge with dialogue data
df_final = df_dialogue.merge(df_outcomes, on="dialogue_id", how="inner")

# Save the merged data
df_final.to_csv("/home/gganeshl/FOLIAGE/datasets/jobinterview/final_dialogues_with_outcomes.csv", index=False)
print("Saved to final_dialogues_with_outcomes.csv")