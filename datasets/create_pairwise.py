import pandas as pd
import json
import os
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np

class PairwiseDatasetSplitter:
    def __init__(self, main_csv_path: str, persuasion_json_path: str):
        """
        Initialize with main dataset CSV and persuasion analysis JSON
        """
        # Load main dataset
        self.main_df = pd.read_csv(main_csv_path)
        
        # Load persuasion analysis
        with open(persuasion_json_path, 'r') as f:
            self.persuasion_data = json.load(f)
        
        # Extract intention mapping from persuasion analysis
        self.intention_mapping = self._create_intention_mapping()
        
        # Define categories
        self.categories = {
            'expressed_donate_did': 'Expressed intention of donating but did not donate',
            'expressed_donate_donated': 'Expressed intention and donated', 
            'no_express_donated': 'Did not express intention but donated',
            'no_express_no_donate': 'Did not express intention and did not donate',
            'unclear_donated': 'Unclear but donated',
            'unclear_no_donate': 'Unclear but did not donate'
        }
        
        # Split data into categories
        self.categorized_data = self._categorize_data()
        
    def _create_intention_mapping(self) -> Dict[str, str]:
        """
        Create mapping from dialogue_id to stated intention from persuasion analysis
        """
        intention_map = {}
        
        detailed_results = self.persuasion_data.get('detailed_results', [])
        
        for result in detailed_results:
            if 'error' not in result:
                dialogue_id = result.get('dialogue_id')
                stated_intention = result.get('stated_intention')
                
                if dialogue_id and stated_intention:
                    intention_map[dialogue_id] = stated_intention
        
        return intention_map
    
    def _categorize_data(self) -> Dict[str, pd.DataFrame]:
        """
        Split main dataset into 6 categories based on intention and actual donation
        """
        categorized = {cat: [] for cat in self.categories.keys()}
        
        # Group by dialogue_id to get conversation-level info
        dialogue_groups = self.main_df.groupby('dialogue_id')
        
        for dialogue_id, group in dialogue_groups:
            # Get stated intention from persuasion analysis
            stated_intention = self.intention_mapping.get(dialogue_id, 'unclear')
            
            # Get actual donation status (use the first non-null value)
            donation_made = group['donation_made'].dropna()
            actual_donated = donation_made.iloc[0] if len(donation_made) > 0 else False
            actual_donated = bool(actual_donated)  # Ensure boolean
            
            # Categorize based on intention and actual donation
            if stated_intention == 'donate':
                if actual_donated:
                    category = 'expressed_donate_donated'
                else:
                    category = 'expressed_donate_did'
            elif stated_intention == 'not_donate':
                if actual_donated:
                    category = 'no_express_donated'
                else:
                    category = 'no_express_no_donate'
            else:  # unclear intention
                if actual_donated:
                    category = 'unclear_donated'
                else:
                    category = 'unclear_no_donate'
            
            # Add all utterances from this conversation to the category
            categorized[category].append(group)
        
        # Convert lists to DataFrames
        final_categorized = {}
        for category, data_list in categorized.items():
            if data_list:
                final_categorized[category] = pd.concat(data_list, ignore_index=True)
            else:
                final_categorized[category] = pd.DataFrame()
        
        return final_categorized
    
    def print_category_summary(self):
        """
        Print summary of how many conversations fall into each category
        """
        print("="*80)
        print("DATASET CATEGORIZATION SUMMARY")
        print("="*80)
        
        total_conversations = 0
        total_utterances = 0
        
        for cat_key, cat_name in self.categories.items():
            cat_df = self.categorized_data[cat_key]
            
            if len(cat_df) > 0:
                unique_dialogues = cat_df['dialogue_id'].nunique()
                total_utterances_cat = len(cat_df)
                
                total_conversations += unique_dialogues
                total_utterances += total_utterances_cat
                
                print(f"\n{cat_name}:")
                print(f"  • Conversations: {unique_dialogues}")
                print(f"  • Total utterances: {total_utterances_cat}")
                print(f"  • Avg utterances per conversation: {total_utterances_cat/unique_dialogues:.1f}")
            else:
                print(f"\n{cat_name}:")
                print(f"  • Conversations: 0")
                print(f"  • Total utterances: 0")
        
        print(f"\nOVERALL TOTALS:")
        print(f"  • Total conversations: {total_conversations}")
        print(f"  • Total utterances: {total_utterances}")
        
        # Check coverage
        original_conversations = self.main_df['dialogue_id'].nunique()
        original_utterances = len(self.main_df)
        
        print(f"\nCOVERAGE CHECK:")
        print(f"  • Original conversations: {original_conversations}")
        print(f"  • Categorized conversations: {total_conversations}")
        print(f"  • Coverage: {total_conversations/original_conversations:.1%}")
        
        print(f"  • Original utterances: {original_utterances}")
        print(f"  • Categorized utterances: {total_utterances}")
        print(f"  • Coverage: {total_utterances/original_utterances:.1%}")
    
    def create_pairwise_datasets(self, output_dir: str = 'pairwise_datasets'):
        """
        Create all 15 pairwise combinations (6C2) of the categories
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get category keys that have data
        available_categories = [cat for cat in self.categories.keys() 
                              if len(self.categorized_data[cat]) > 0]
        
        if len(available_categories) < 2:
            print("Error: Need at least 2 categories with data to create pairwise datasets")
            return
        
        # Generate all pairwise combinations
        pairwise_combinations = list(combinations(available_categories, 2))
        
        print(f"\n" + "="*80)
        print(f"CREATING {len(pairwise_combinations)} PAIRWISE DATASETS")
        print("="*80)
        
        dataset_info = []
        
        for i, (cat1, cat2) in enumerate(pairwise_combinations, 1):
            # Get data for both categories
            data1 = self.categorized_data[cat1].copy()
            data2 = self.categorized_data[cat2].copy()
            
            if len(data1) == 0 or len(data2) == 0:
                print(f"Skipping {cat1} vs {cat2}: One category has no data")
                continue
            
            # Add binary labels for classification
            data1['binary_label'] = 0  # First category gets label 0
            data2['binary_label'] = 1  # Second category gets label 1
            
            # Add category information
            data1['category'] = cat1
            data2['category'] = cat2
            
            # Combine datasets
            pairwise_df = pd.concat([data1, data2], ignore_index=True)
            
            # Create descriptive filename
            cat1_short = cat1.replace('_', '')
            cat2_short = cat2.replace('_', '')
            filename = f"pairwise_{i:02d}_{cat1_short}_vs_{cat2_short}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save to CSV
            pairwise_df.to_csv(filepath, index=False)
            
            # Track dataset info
            dataset_info.append({
                'file': filename,
                'category_1': cat1,
                'category_1_desc': self.categories[cat1],
                'category_2': cat2, 
                'category_2_desc': self.categories[cat2],
                'conversations_cat1': data1['dialogue_id'].nunique(),
                'conversations_cat2': data2['dialogue_id'].nunique(),
                'utterances_cat1': len(data1),
                'utterances_cat2': len(data2),
                'total_conversations': pairwise_df['dialogue_id'].nunique(),
                'total_utterances': len(pairwise_df),
                'balance_ratio': min(len(data1), len(data2)) / max(len(data1), len(data2))
            })
            
            print(f"\n{i:2d}. {filename}")
            print(f"    {self.categories[cat1][:50]}...")
            print(f"    vs")
            print(f"    {self.categories[cat2][:50]}...")
            print(f"    • Cat1: {data1['dialogue_id'].nunique()} conv, {len(data1)} utt")
            print(f"    • Cat2: {data2['dialogue_id'].nunique()} conv, {len(data2)} utt")
            print(f"    • Balance: {dataset_info[-1]['balance_ratio']:.3f}")
        
        # Save dataset information summary
        info_df = pd.DataFrame(dataset_info)
        info_filepath = os.path.join(output_dir, 'pairwise_datasets_summary.csv')
        info_df.to_csv(info_filepath, index=False)
        
        # Create detailed README
        self._create_readme(output_dir, dataset_info)
        
        print(f"\n" + "="*80)
        print(f"PAIRWISE DATASETS CREATED SUCCESSFULLY")
        print(f"Output directory: {output_dir}")
        print(f"Total pairwise datasets: {len(dataset_info)}")
        print(f"Summary file: {info_filepath}")
        print("="*80)
        
        return dataset_info
    
    def _create_readme(self, output_dir: str, dataset_info: List[Dict]):
        """
        Create a detailed README file explaining the datasets
        """
        readme_path = os.path.join(output_dir, 'README.md')
        
        with open(readme_path, 'w') as f:
            f.write("# Pairwise Persuasion Datasets\n\n")
            f.write("This directory contains 15 pairwise datasets for training binary classifiers on persuasion categories.\n\n")
            
            f.write("## Categories\n\n")
            for i, (cat_key, cat_desc) in enumerate(self.categories.items(), 1):
                f.write(f"{i}. **{cat_key}**: {cat_desc}\n")
            
            f.write("\n## Dataset Structure\n\n")
            f.write("Each CSV file contains:\n")
            f.write("- All original columns from the main dataset\n")
            f.write("- `binary_label`: 0 for first category, 1 for second category\n")
            f.write("- `category`: The specific category name\n\n")
            
            f.write("## Pairwise Combinations\n\n")
            for info in dataset_info:
                f.write(f"### {info['file']}\n")
                f.write(f"- **Category 1 (Label 0)**: {info['category_1_desc']}\n")
                f.write(f"  - Conversations: {info['conversations_cat1']}\n")
                f.write(f"  - Utterances: {info['utterances_cat1']}\n")
                f.write(f"- **Category 2 (Label 1)**: {info['category_2_desc']}\n")
                f.write(f"  - Conversations: {info['conversations_cat2']}\n")
                f.write(f"  - Utterances: {info['utterances_cat2']}\n")
                f.write(f"- **Balance Ratio**: {info['balance_ratio']:.3f}\n")
                f.write(f"- **Total**: {info['total_conversations']} conversations, {info['total_utterances']} utterances\n\n")
            
            f.write("## Usage\n\n")
            f.write("Each dataset can be used to train a binary classifier to distinguish between two specific persuasion behavior patterns. The binary_label column serves as the target variable.\n\n")
            f.write("## Files\n\n")
            f.write("- `pairwise_*.csv`: Individual pairwise datasets\n")
            f.write("- `pairwise_datasets_summary.csv`: Summary statistics for all datasets\n")
            f.write("- `README.md`: This documentation\n")
    
    def analyze_class_balance(self) -> pd.DataFrame:
        """
        Analyze class balance across all pairwise combinations
        """
        balance_analysis = []
        
        available_categories = [cat for cat in self.categories.keys() 
                              if len(self.categorized_data[cat]) > 0]
        
        for cat1, cat2 in combinations(available_categories, 2):
            data1 = self.categorized_data[cat1]
            data2 = self.categorized_data[cat2]
            
            if len(data1) > 0 and len(data2) > 0:
                balance_ratio = min(len(data1), len(data2)) / max(len(data1), len(data2))
                
                balance_analysis.append({
                    'category_1': cat1,
                    'category_2': cat2,
                    'utterances_1': len(data1),
                    'utterances_2': len(data2),
                    'conversations_1': data1['dialogue_id'].nunique(),
                    'conversations_2': data2['dialogue_id'].nunique(),
                    'balance_ratio': balance_ratio,
                    'imbalance_severity': 'Severe' if balance_ratio < 0.3 else 'Moderate' if balance_ratio < 0.7 else 'Good'
                })
        
        return pd.DataFrame(balance_analysis)

def main():
    # File paths - UPDATE THESE
    MAIN_CSV = "datasets/p4g/final/ratio_1.csv"  # Your main dataset CSV
    PERSUASION_JSON = "persuasion_analysis_results.json"  # Your persuasion analysis JSON
    OUTPUT_DIR = "pairwise_datasets"  # Output directory for pairwise datasets
    
    print("Initializing Pairwise Dataset Splitter...")
    splitter = PairwiseDatasetSplitter(MAIN_CSV, PERSUASION_JSON)
    
    # Print category summary
    splitter.print_category_summary()
    
    # Analyze class balance
    print("\n" + "="*80)
    print("CLASS BALANCE ANALYSIS")
    print("="*80)
    balance_df = splitter.analyze_class_balance()
    
    if len(balance_df) > 0:
        print(f"\nBalance Ratio Distribution:")
        print(f"Good (>0.7): {len(balance_df[balance_df['balance_ratio'] > 0.7])}")
        print(f"Moderate (0.3-0.7): {len(balance_df[(balance_df['balance_ratio'] >= 0.3) & (balance_df['balance_ratio'] <= 0.7)])}")
        print(f"Severe (<0.3): {len(balance_df[balance_df['balance_ratio'] < 0.3])}")
        
        print(f"\nMost Imbalanced Pairs:")
        worst_balanced = balance_df.nsmallest(3, 'balance_ratio')
        for _, row in worst_balanced.iterrows():
            print(f"  {row['category_1']} vs {row['category_2']}: {row['balance_ratio']:.3f}")
        
        print(f"\nBest Balanced Pairs:")
        best_balanced = balance_df.nlargest(3, 'balance_ratio')
        for _, row in best_balanced.iterrows():
            print(f"  {row['category_1']} vs {row['category_2']}: {row['balance_ratio']:.3f}")
    
    # Create pairwise datasets
    dataset_info = splitter.create_pairwise_datasets(OUTPUT_DIR)
    
    print(f"\n🎉 Successfully created {len(dataset_info)} pairwise datasets!")
    print(f"Check the '{OUTPUT_DIR}' directory for all files.")

if __name__ == "__main__":
    main()