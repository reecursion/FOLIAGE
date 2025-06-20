import pandas as pd
import numpy as np
import json
import os
import random
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict as ddict
from scipy.stats import pearsonr, spearmanr
from pprint import pprint


def get_p4g_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    infos       = ['none', 'local', 'global_scd', 'global_scm', 'global_traditional', 'both_scd', 'both_scm', 'both_traditional']

    results_dir = f'../src/{args.method}/results/{args.dataset}/'

    # for seed in seeds:

    results_dict = ddict(list)


    for info in infos:

        results_dict['Info'].append(info)
        for frac in fracs:

            mf1_arr = []
            for seed in seeds:

                # load the csv file
                csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_classification_ratio_{frac}_{info}_predictions.csv'

                try:
                    df       = pd.read_csv(csv_file)

                    y_trues = df['label'].values
                    y_preds = df['predicted_label'].values


                    # print(f'
                    # Loaded {csv_file} with shape {df.shape}')
                except Exception as e:
                    print(f'Error loading {csv_file}: {e}')
                    continue

                # dialogue_id,fold,text,label,predicted_label,correct,confidence_score

               
                mf1     = f1_score(y_trues, y_preds, average='macro')
                mf1_arr.append(100*mf1)
            
            # calculate the mean and std of the mf1
            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}' 
            
            results_dict[frac].append(result)
            print(f'Info: {info}, frac: {frac}, mf1: {result}')

    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-{args.dataset}-agg_perfs.csv', index=False)
    



def get_cd_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['11611', '10623', '42']

    infos       = ['none', 'local', 'global_scd', 'global_scm', 'global_traditional', 'both_scd', 'both_scm', 'both_traditional']

    if args.method == 'sft':
        results_dir = f'../src/{args.method}/results/{args.dataset}/'

    # elif args.method == 'baseline':
    #     results_dir = f''

    # for seed in seeds:

    results_dict = ddict(list)


    for info in infos:

        results_dict['Info'].append(info)
        for frac in fracs:

            mf1_arr = []
            for seed in seeds:

                # load the csv file
                csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_classification_ratio_{frac}_{info}_predictions.csv'

                try:
                    df       = pd.read_csv(csv_file)

                    y_trues = df['label'].values
                    y_preds = df['predicted_label'].values


                    # print(f'
                    # Loaded {csv_file} with shape {df.shape}')
                except Exception as e:
                    print(f'Error loading {csv_file}: {e}')
                    continue

                # dialogue_id,fold,text,label,predicted_label,correct,confidence_score

               
                mf1     = f1_score(y_trues, y_preds, average='macro')
                mf1_arr.append(100*mf1)
            
            # calculate the mean and std of the mf1
            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}' 
            
            results_dict[frac].append(result)
            print(f'Info: {info}, frac: {frac}, mf1: {result}')

    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-{args.dataset}-agg_perfs.csv', index=False)
    



def get_cb_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    infos       = ['none', 'local', 'global_scd', 'global_scm', 'global_traditional', 'both_scd', 'both_scm', 'both_traditional']

    results_dir = f'../src/{args.method}/results/{args.dataset}/'

    # for seed in seeds:


    results_dict = ddict(list)

    metrics      = ['rmse', 'corr', 'nmse', 'rmse_price' ]
    # metrics       = ['rmse']

    correct_files = 0
    tot_files     = 0

    for metric in metrics:
        for info in infos:
            results_dict['Info'].append(info)
            results_dict['Metric'].append(metric)

            for frac in fracs:

                rmse_arr = []
                corr_arr = []
                nmse_arr = []
                rmse_price_arr = []

                for seed in seeds:
                    tot_files += 1

                    # load the csv file
                    csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_ratio_{frac}_{info}_predictions.csv'

                    try:
                        df       = pd.read_csv(csv_file)
                        # print(f'
                        # Loaded {csv_file} with shape {df.shape}')
                    except Exception as e:
                        print(f'Error loading {csv_file}')
                        continue
                    
                    if len(df) == 0:
                        print(f'Empty dataframe: {csv_file}')
                        continue
                    
                    correct_files += 1

                    # dialogue_id,fold,prompt,generated_text,buyer_target,seller_target,sale_price,predicted_final_price,success_sale,success_predicted,normalized_squared_error
                    try:
                        y_true_success = df['success_sale'].values
                        y_pred_success = df['success_predicted'].values

                        # import pdb; pdb.set_trace()

                        rmse           = np.sqrt(np.mean((y_true_success - y_pred_success)**2))
                        # pearson correlation
                        pearson_corr   = pearsonr(y_true_success, y_pred_success)[0]

                        rmse_arr.append(rmse)
                        corr_arr.append(pearson_corr)

                        
                        seller_targets = df['seller_target'].values
                        buyer_targets  = df['buyer_target'].values
                        sale_prices    = df['sale_price'].values
                        pred_prices    = df['predicted_final_price'].values


                        nmse_tot        = []
                        rmse_price_tot  = []

                        for i in range(len(seller_targets)):
                            
                            nmse_val = (sale_prices[i] - pred_prices[i])**2 / (sale_prices[i])**2
                            nmse_tot.append(nmse_val)

                            rmse_price_val = (sale_prices[i] - pred_prices[i])**2
                            rmse_price_tot.append(rmse_price_val)
                        
                        nmse_val = np.sqrt(np.mean(nmse_tot))
                        nmse_arr.append(nmse_val)

                        rmse_price_val = np.sqrt(np.mean(rmse_price_tot))
                        rmse_price_arr.append(rmse_price_val)
                        

                    except Exception as e:
                        print(f'Error calculating metrics: {e}')
                        import pdb; pdb.set_trace()
                    

                    # print(f'csv_file: {csv_file}, rmse {rmse}, corr {pearson_corr}, nmse {nmse_val}, rmse_price {rmse_price_val}')
                    

                
                # calculate the mean and std of the mf1
                rmse_mean = np.mean(rmse_arr)
                rmse_std  = np.std(rmse_arr)
                corr_mean = np.mean(corr_arr)
                corr_std  = np.std(corr_arr)
                nmse_mean = np.mean(nmse_arr)
                nmse_std  = np.std(nmse_arr)
                rmse_price_mean = np.mean(rmse_price_arr)
                rmse_price_std  = np.std(rmse_price_arr)

                if metric == 'rmse':
                    result  = f'{rmse_mean:.2f} ± {rmse_std:.2f}'
                elif metric == 'corr':
                    result  = f'{corr_mean:.2f} ± {corr_std:.2f}'
                elif metric == 'nmse':
                    result  = f'{nmse_mean:.2f} ± {nmse_std:.2f}'
                elif metric == 'rmse_price':
                    result  = f'{rmse_price_mean:.2f} ± {rmse_price_std:.2f}'
                
                
                results_dict[frac].append(result)

    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-{args.dataset}-agg_perfs.csv', index=False)
    
    print(f'Correct files: {correct_files}/{tot_files} = {correct_files/tot_files:.2f}')


def get_casino_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    infos       = ['none', 'local', 'global_scd', 'global_scm', 'global_traditional', 'both_scd', 'both_scm', 'both_traditional']

    results_dir = f'../src/{args.method}/results/{args.dataset}/'

    correct_files = 0
    tot_files     = 0

    results_dict = ddict(list)
    
    empty_cases     = ddict(lambda: ddict(int))

    for info in infos:
        results_dict['Info'].append(info)

        for frac in fracs:

            mse_arr = []

            for seed in seeds:
                tot_files += 1

                # load the csv file
                csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_ratio_{frac}_{info}_predictions.csv'

                try:
                    df       = pd.read_csv(csv_file)
                except Exception as e:
                    print(f'Error loading {csv_file}: {e}')
                    continue
                
                if len(df) == 0:
                    print(f'Empty dataframe: {csv_file}')
                    continue
                
                correct_files += 1

                # dialogue_id,fold,prompt,generated_text,agent1_food,agent1_water,agent1_firewood,agent2_food,agent2_water,agent2_firewood,predicted_agent1_food,predicted_agent1_water,predicted_agent1_firewood,predicted_agent2_food,predicted_agent2_water,predicted_agent2_firewood,true_agent1_utility,true_agent2_utility,predicted_agent1_utility,predicted_agent2_utility,utility_mse

                mse_vals = df['utility_mse'].values

                # remove mse_vals that are NaN

                mean_mse = np.mean(mse_vals)

                if mean_mse != mean_mse:
                    new_mse_vals = mse_vals[~np.isnan(mse_vals)]
                    mean_mse = np.mean(new_mse_vals)

                    empty_cases[info][frac] += len(mse_vals) - len(new_mse_vals)                
                
                else:
                    empty_cases[info][frac] += 0
                
                
                # print(f'csv_file: {csv_file}, mse_vals: {mean_mse}')

                mse_arr.append(mean_mse)

            
            # calculate the mean and std of the mf1
            mse_mean = np.mean(mse_arr)
            mse_std  = np.std(mse_arr)
            
            
            result  = f'{mse_mean:.2f} ± {mse_std:.2f}'
            
            results_dict[frac].append(result)

    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-{args.dataset}-agg_perfs.csv', index=False)
    
    print(f'Correct files: {correct_files}/{tot_files} = {correct_files/tot_files:.2f}')

    pprint(empty_cases)


if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='p4g')
    parser.add_argument('--step',    type=str, required=True)
    parser.add_argument('--method',  type=str, default='sft')
    
    
    args = parser.parse_args()


    if args.step == 'get_perf':
        
        if args.dataset == 'p4g':
            get_p4g_perfs()
        
        elif args.dataset == 'cb':
            get_cb_perfs()

        elif args.dataset == 'casino':
            get_casino_perfs()
        
        elif args.dataset == 'cd':
            get_cd_perfs()