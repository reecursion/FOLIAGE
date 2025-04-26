import pandas as pd

# Load the CSV file
df = pd.read_csv('/home/gganeshl/FOLIAGE/datasets/jobinterview/utils/utterance_intentions.csv')
unique_dialogue_ids = df['dialogue_id'].nunique()
print(f"Number of unique dialogue IDs, before filtering: {unique_dialogue_ids}")


df = pd.read_csv('/home/gganeshl/FOLIAGE/datasets/jobinterview/utterance_intentions.csv')
unique_dialogue_ids = df['dialogue_id'].nunique()
print(f"Number of unique dialogue IDs, after filtering: {unique_dialogue_ids}")