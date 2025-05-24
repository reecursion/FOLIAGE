import pandas as pd
import numpy as np
import json
import os
import random
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score



def get_seedwise_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    infos       = ['none', 'local', 'global_scd', 'global_scm', 'global_traditional', 'both_scd', 'both_scm', 'both_traditional']

    results_dir = f'../src/sft/results/{args.dataset}/'

    # for seed in seeds:

    results_dict = ddict(list)


    for info in infos:

        for frac in fracs:

            for seed in seeds:

                # load the csv file
                csv_file = f'{results_dir}/seed_{seed}/{dataset}_classification_ratio_{frac}_{info}_predictions.csv'

                df       = pd.read_csv(csv_file)

                # dialogue_id,fold,text,label,predicted_label,correct,confidence_score

                y_trues = df['label'].values
                y_preds = df['predicted_label'].values

                mf1     = f1_score(y_trues, y_preds, average='macro')



                


    





if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='p4g')
    parser.add_argument('--step',    type=str, required=True)
    
    
    args = parser.parse_args()


    if args.step == 'get_perf':
        get_seedwise_perfs()