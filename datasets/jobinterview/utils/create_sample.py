import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/home/gganeshl/FOLIAGE/datasets/jobinterview/utils/final_dialogues_with_outcomes.csv"
df = pd.read_csv(file_path)

# Create uniformly spaced buckets based on worker scores
min_score = df['score_worker'].min()
max_score = df['score_worker'].max()

# Create 10 uniformly spaced bins
bins = np.linspace(min_score, max_score, 11)
labels = [f"Bucket {i+1}" for i in range(10)]

# Add a bucket column to the dataframe
df['worker_score_bucket'] = pd.cut(df['score_worker'], bins=bins, labels=labels, include_lowest=True)

# Now group by dialogue ID and pick one score per dialogue
# Assuming the 'score_worker' is the same per dialogue, or take the first if varies
dialogue_scores = df.groupby('dialogue_id').first().reset_index()[['dialogue_id', 'score_worker', 'worker_score_bucket']]

# Create a dictionary to store sampled dialogue IDs per bucket
sampled_dialogue_ids = {}

# Randomly sample dialogue IDs (not rows)
for bucket in labels:
    bucket_dialogues = dialogue_scores[dialogue_scores['worker_score_bucket'] == bucket]
    if len(bucket_dialogues) > 25:
        sampled_dialogue_ids[bucket] = bucket_dialogues.sample(n=35, random_state=44)['dialogue_id'].tolist()
    else:
        sampled_dialogue_ids[bucket] = bucket_dialogues['dialogue_id'].tolist()
    print(f"{bucket}: {len(sampled_dialogue_ids[bucket])} dialogues sampled (out of {len(bucket_dialogues)})")

# Collect all rows corresponding to the sampled dialogue IDs
sampled_rows = df[df['dialogue_id'].isin([did for ids in sampled_dialogue_ids.values() for did in ids])]

# Save the sampled dataset
sampled_file_path = "/home/gganeshl/FOLIAGE/datasets/jobinterview/utterance_intentions.csv"
sampled_rows.to_csv(sampled_file_path, index=False)

# Display statistics
print(f"\nTotal sampled dialogues: {len(set(sampled_rows['dialogue_id']))}")
print(f"Total sampled rows: {len(sampled_rows)}")
sampled_dialogue_stats = sampled_rows.groupby('worker_score_bucket')['score_worker'].agg(['count', 'min', 'max', 'mean'])
print("\nWorker score statistics by bucket (sampled dialogues):")
print(sampled_dialogue_stats)

# For comparing original vs sampled dataset distribution
# Get dialogue-level statistics for original dataset
original_dialogue_counts = dialogue_scores['worker_score_bucket'].value_counts().sort_index()
original_dialogue_percentage = (original_dialogue_counts / original_dialogue_counts.sum() * 100).round(2)

# Get dialogue-level statistics for sampled dataset
sampled_dialogue_ids_flat = [did for ids in sampled_dialogue_ids.values() for did in ids]
sampled_dialogue_scores = dialogue_scores[dialogue_scores['dialogue_id'].isin(sampled_dialogue_ids_flat)]
sampled_dialogue_counts = sampled_dialogue_scores['worker_score_bucket'].value_counts().sort_index()
sampled_dialogue_percentage = (sampled_dialogue_counts / sampled_dialogue_counts.sum() * 100).round(2)

# Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Original Count': original_dialogue_counts,
    'Original %': original_dialogue_percentage,
    'Sampled Count': sampled_dialogue_counts,
    'Sampled %': sampled_dialogue_percentage
})

print("\nOriginal vs Sampled Dataset Distribution (Dialogue ID wise):")
print(comparison_df)

# Compare final score statistics
print("\nOriginal Dataset Score Statistics:")
print(dialogue_scores['score_worker'].describe())

print("\nSampled Dataset Score Statistics:")
print(sampled_dialogue_scores['score_worker'].describe())