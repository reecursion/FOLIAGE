import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import argparse
import re

def extract_price(response):
    """Extract price from FINAL_PRICE format in the response column."""
    price_match = re.search(r'FINAL_PRICE: \$(\d+)', response)
    if price_match:
        return float(price_match.group(1))
    return np.nan

def calculate_scores(file_path):
    """Process a single CSV file and print metrics."""
    try:
        df = pd.read_csv(file_path)
        print(f"Original length of dataset: {len(df)}")
        
        if 'predicted_final_price' not in df.columns and 'response' in df.columns:
            df['predicted_final_price'] = df['response'].apply(extract_price)
            print("Extracted predicted prices from response column")

        original_len = len(df)
        df = df.dropna(subset=['sale_price', 'predicted_final_price', 'buyer_target', 'seller_target'])
        print(f"Dropped {original_len - len(df)} rows with NaN values")
        
        if len(df) == 0:
            print(f"No valid data in {file_path} after NaN filtering")
            return
            
        # Handle both cases: buyer_target < seller_target and buyer_target > seller_target
        valid_mask = ((df['sale_price'] >= df['buyer_target']) & (df['sale_price'] <= df['seller_target'])) | \
                     ((df['sale_price'] <= df['buyer_target']) & (df['sale_price'] >= df['seller_target']))
        
        original_len = len(df)
        df = df[valid_mask]
        print(f"Removed {original_len - len(df)} rows with sale prices outside buyer-seller range")
        
        if len(df) == 0:
            print(f"No valid data in {file_path} after sale price range filtering")
            return

        df['r_sale'] = (df['sale_price'] - df['buyer_target']) / (df['seller_target'] - df['buyer_target'])
        df['r_predicted'] = (df['predicted_final_price'] - df['buyer_target']) / (df['seller_target'] - df['buyer_target'])

        price_mse = mean_squared_error(df['sale_price'], df['predicted_final_price'])
        price_rmse = np.sqrt(price_mse)

        df["normalized_squared_error"] = ((df["sale_price"] - df["predicted_final_price"]) ** 2) / df["sale_price"]
        price_nmse = df["normalized_squared_error"].mean()
        price_pearson, _ = pearsonr(df['sale_price'], df['predicted_final_price'])

        r_mse = mean_squared_error(df['r_sale'], df['r_predicted'])
        r_rmse = np.sqrt(r_mse)
        r_pearson, _ = pearsonr(df['r_sale'], df['r_predicted'])

        # Print results
        print("\n----- RESULTS -----")
        print(f"Sample size: {len(df)}")
        print(f"Price RMSE: {price_rmse:.3f}")
        print(f"Price NMSE: {price_nmse:.3f}")
        print(f"Price Pearson correlation: {price_pearson:.3f}")
        print(f"Success Score RMSE: {r_rmse:.3f}")
        print(f"Success Score Pearson correlation: {r_pearson:.3f}")
        
        return {
            'price_rmse': price_rmse,
            'price_nmse': price_nmse,
            'price_pearson': price_pearson,
            'r_rmse': r_rmse,
            'r_pearson': r_pearson,
            'sample_size': len(df)
        }

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate negotiation prediction scores from a CSV file')
    parser.add_argument('--input_file', help='Path to the input CSV file')
    args = parser.parse_args()
    
    print(f"Processing file: {args.input_file}")
    calculate_scores(args.input_file)

if __name__ == "__main__":
    main()