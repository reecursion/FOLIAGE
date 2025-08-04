import pandas as pd

# Load the CSV file
path = "/home/gganeshl/FOLIAGE/datasets/p4g/final/ratio_0.5.csv"
df = pd.read_csv(path)

# Print unique values and their counts in the donation_made column
if 'donation_made' in df.columns:
    # Get value counts
    donation_counts = df['donation_made'].value_counts()
    
    # Get percentages
    donation_percentages = df['donation_made'].value_counts(normalize=True) * 100
    
    # Print results with both counts and percentages (rounded to 2 decimal places)
    print("Unique values in 'donation_made' column and their counts:")
    for value, count in donation_counts.items():
        percentage = donation_percentages[value]
        print(f"Value: {value}, Count: {count}, Percentage: {percentage:.2f}%")
    
    # Additional statistics
    print("\nSummary statistics:")
    print(f"Total records: {len(df)}")
    print(f"Number of unique values: {len(donation_counts)}")
else:
    print("The 'donation_made' column doesn't exist in this dataset.")