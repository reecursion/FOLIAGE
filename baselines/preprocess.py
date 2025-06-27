import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import json
import random
from collections import defaultdict as ddict, Counter


def split_ids2folds(ids, num_folds):

    fold_dict = ddict(list)

    random.shuffle(ids)

    for idx, id in enumerate(ids):
        fold = idx % num_folds
        fold_dict[fold].append(id)

    return fold_dict

def preprocess_wiki(args):

    # dialogue_id,turn_id,utterance_idx,speaker,utterance,personal_attack,intention,traditional_summary,scm_summary,scd_summary

    fracwise_dialogs    = ddict(list)
    df = pd.read_csv(f'datasets/p4g/final/ratio_{args.frac}.csv')
    # df.rename(columns={'politeness_theory_stage2_summary': 'politeness_summary'}, inplace=True)
    uniq_conversations  = df['dialogue_id'].unique()

    foldwise_conv_ids   = split_ids2folds(uniq_conversations, args.num_folds)

    test_foldwise_data   = ddict(list)
    train_foldwise_data  = ddict(list)

    for frac in [1]:
        
        df = pd.read_csv(f'datasets/p4g/final/ratio_{frac}.csv')
        # df.rename(columns={'politeness_theory_stage2_summary': 'politeness_summary'}, inplace=True)
        uniq_conversations  = df['dialogue_id'].unique()

        for fold in range(args.num_folds):

            test_conv_data  = []
            train_conv_data = []

            for dialog_id in uniq_conversations:
                
                curr_data       = {'dialogue_id': dialog_id, \
                        'speakers': [],\
                        'utterances': [],\
                        'intentions': [],\
                        'traditional_summary':"", \
                        'scm_summary':"", \
                        'scd_summary':""}

                curr_df = df[df['dialogue_id'] == dialog_id]
                dialog_history = []

                for idx, row in curr_df.iterrows():
                    curr_data['speakers'].append(row['speaker'])
                    curr_data['utterances'].append(row['utterance'])
                    curr_data['intentions'].append(row['intention'])

                    if curr_data['traditional_summary'] == "":
                        curr_data['traditional_summary'] = row['traditional_summary']
                    else:
                        assert row['traditional_summary'] == curr_data['traditional_summary']
                    
                    if curr_data['scm_summary'] == "":
                        curr_data['scm_summary'] = row['scm_summary']
                    else:
                        assert row['scm_summary'] == curr_data['scm_summary']
                    
                    if curr_data['scd_summary'] == "":
                        curr_data['scd_summary'] = row['scd_summary']
                    else:
                        assert row['scd_summary'] == curr_data['scd_summary']

                curr_data['binary_score']  = row["donation_made"]

                if dialog_id in foldwise_conv_ids[fold]:
                    test_conv_data.append(curr_data)
                    test_foldwise_data[fold].append(curr_data)

                else:
                    train_conv_data.append(curr_data)
                    train_foldwise_data[fold].append(curr_data)
            

            with open(f'baselines/data/{args.dataset}/processed/RAT_{frac}_{fold}_train.json', 'w') as f:
                json.dump(train_conv_data, f, indent=4)
            
            with open(f'baselines/data/{args.dataset}/processed/RAT_{frac}_{fold}_test.json', 'w') as f:
                json.dump(test_conv_data, f, indent=4)
        
            # print("Length of train data: ", len(train_conv_data))
            # print("Length of test data: ", len(test_conv_data))
    
    for fold in range(args.num_folds):

        with open(f'baselines/data/{args.dataset}/processed/RAT_ALL_{fold}_train.json', 'w') as f:
            json.dump(train_foldwise_data[fold], f, indent=4)
        
        with open(f'baselines/data/{args.dataset}/processed/RAT_ALL_{fold}_test.json', 'w') as f:
            json.dump(test_foldwise_data[fold], f, indent=4)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--dataset', type=str, default='p4g', help='Path to the data directory.')
    parser.add_argument('--step',    type=str, default='prepare')
    parser.add_argument('--frac',    type=str, default=0.25, help='Fraction of data to use.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation.')
    
    args = parser.parse_args()

    preprocess_wiki(args)