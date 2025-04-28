import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved dataset
file_path = "/home/gganeshl/FOLIAGE/datasets/jobinterview/utils/final_dialogues_with_outcomes.csv"
df = pd.read_csv(file_path)

# Create uniformly spaced buckets based on worker scores
# First, determine the min and max values of the worker scores
min_score = df['score_worker'].min()
max_score = df['score_worker'].max()

# Create 10 uniformly spaced bins
bins = np.linspace(min_score, max_score, 11)
labels = [f"Bucket {i+1}" for i in range(10)]

# Add a bucket column to the dataframe
df['worker_score_bucket'] = pd.cut(df['score_worker'], bins=bins, labels=labels, include_lowest=True)

# Create a dictionary to store the sampled data from each bucket
sampled_data = {}

# Randomly sample 25 dialogues from each bucket (or all if fewer than 25)
for bucket in labels:
    bucket_df = df[df['worker_score_bucket'] == bucket]
    if len(bucket_df) > 25:
        sampled_data[bucket] = bucket_df.sample(n=35, random_state=44)
    else:
        sampled_data[bucket] = bucket_df
    print(f"{bucket}: {len(sampled_data[bucket])} samples (out of {len(bucket_df)} available)")

# Combine all the sampled data into a single dataframe
sampled_df = pd.concat(sampled_data.values())

# Save the sampled dataset
sampled_file_path = "/home/gganeshl/FOLIAGE/datasets/jobinterview/utterance_intentions.csv"
sampled_df.to_csv(sampled_file_path, index=False)

# Display statistics about the sampled data
print(f"\nTotal samples: {len(sampled_df)}")
print("\nWorker score statistics by bucket:")
bucket_stats = sampled_df.groupby('worker_score_bucket')['score_worker'].agg(['count', 'min', 'max', 'mean'])
print(bucket_stats)
