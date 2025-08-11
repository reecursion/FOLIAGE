import pandas as pd
import numpy as np
import json
import os
import random
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from collections import defaultdict as ddict
from scipy.stats import pearsonr, spearmanr
from pprint import pprint


### Get the HAN performances ###


def get_p4g_han_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    results_dir = f'../baselines/results/{args.dataset}/'
    infos       = ['None_None', 'intentions_None', 'None_traditional_summary', 'None_scd_summary', 'None_scm_summary', 'intentions_scd_summary', 'intentions_scm_summary', 'intentions_traditional_summary']

    mapping_dict = {
        'None_None': 'UTT',
        'None_traditional_summary': 'UTT + Trad',
        'None_scd_summary': 'UTT + SCD',
        'None_scm_summary': 'UTT + SCM',
        'intentions_None': 'UTT+ INT',
        'intentions_scd_summary': 'UTT+ INT + SCD',
        'intentions_scm_summary': 'UTT + INT + SCM',
        'intentions_traditional_summary': 'UTT + INT + Trad'        
    }


    results_dict = ddict(list)

    all_conversations_dict = ddict(list)

    for info in infos:

        results_dict['Info'].append(mapping_dict[info])
        results_dict['Measure'].append('MF1')

        results_dict['Info'].append(mapping_dict[info])
        results_dict['Measure'].append('BS')

        ### Calculate MF1 ####
        for frac in fracs:

            mf1_arr = []
            bs_arr  = []

            for seed in seeds:
                curr_bs_arr = []

                for fold in range(0, 5):

                    csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_classification_ratio_{frac}_{info}_bert-base-uncased_fold_{fold}_test.csv'

                    try:
                        df       = pd.read_csv(csv_file)

                        y_trues = df['gold_label'].values
                        y_preds = df['predicted_label'].values
                        
                        confidence_scores = df['confidence'].values

                        # these confidence scores are for the predicted class

                        for y_true, y_pred, conf_score in zip(y_trues, y_preds, confidence_scores):
                            if y_pred == y_true:
                                bs = 1 - conf_score
                            else:
                                bs = conf_score

                            curr_bs_arr.append(bs**2)
                        
                        # print(f'
                        # Loaded {csv_file} with shape {df.shape}')
                    except Exception as e:
                        print(f'Error loading {csv_file}: {e}')

                        continue

                # dialogue_id,fold,text,label,predicted_label,correct,confidence_score
                mf1     = f1_score(y_trues, y_preds, average='macro')
                mf1_arr.append(100*mf1)

                bs_score = np.mean(curr_bs_arr)
                bs_arr.append(100*bs_score)

            # calculate the mean and std of the mf1
            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            bs_mean = np.mean(bs_arr)
            bs_std  = np.std(bs_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}' 

            results_dict[frac].append(f'{mf1_mean:.2f} ± {mf1_std:.2f}')
            results_dict[frac].append(f'{bs_mean:.2f} ± {bs_std:.2f}')

            print(f'Info: {info}, frac: {frac}, mf1: {result}')
        
    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-{args.dataset}-agg_perfs.csv', index=False)
    
        


def get_cd_han_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    results_dir = f'../baselines/results/{args.dataset}/'
    infos       = ['None_None', 'intentions_None', 'None_traditional_summary', 'None_scd_summary', 'None_scm_summary', 'intentions_scd_summary', 'intentions_scm_summary', 'intentions_traditional_summary']

    mapping_dict = {
        'None_None': 'UTT',
        'None_traditional_summary': 'UTT + Trad',
        'None_scd_summary': 'UTT + SCD',
        'None_scm_summary': 'UTT + SCM',
        'intentions_None': 'UTT+ INT',
        'intentions_scd_summary': 'UTT+ INT + SCD',
        'intentions_scm_summary': 'UTT + INT + SCM',
        'intentions_traditional_summary': 'UTT + INT + Trad'        
    }

    results_dict = ddict(list)

    for info in infos:

        results_dict['Info'].append(mapping_dict[info])
        results_dict['Measure'].append('MF1')

        results_dict['Info'].append(mapping_dict[info])
        results_dict['Measure'].append('BS')

        for frac in fracs:

            mf1_arr = []
            bs_arr  = []

            for seed in seeds:
                curr_bs_arr = []

                for fold in range(0, 5):

                    
                    csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_classification_ratio_{frac}_{info}_bert-base-uncased_fold_{fold}_test.csv'

                    try:
                        df       = pd.read_csv(csv_file)

                        y_trues = df['gold_label'].values
                        y_preds = df['predicted_label'].values

                        confidence_scores = df['confidence'].values

                        # these confidence scores are for the predicted class

                        for y_true, y_pred, conf_score in zip(y_trues, y_preds, confidence_scores):
                            if y_pred == y_true:
                                bs = 1 - conf_score
                            else:
                                bs = conf_score

                            curr_bs_arr.append(bs**2)


                        # print(f'
                        # Loaded {csv_file} with shape {df.shape}')
                    except Exception as e:
                        print(f'Error loading {csv_file}: {e}')

                        continue

                # dialogue_id,fold,text,label,predicted_label,correct,confidence_score

               
                mf1     = f1_score(y_trues, y_preds, average='macro')
                mf1_arr.append(100*mf1)

                bs_score = np.mean(curr_bs_arr)
                bs_arr.append(100*bs_score)
            
            # calculate the mean and std of the mf1
            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}'             
            results_dict[frac].append(result)

            bs_mean = np.mean(bs_arr)
            bs_std  = np.std(bs_arr)

            results_dict[frac].append(f'{bs_mean:.2f} ± {bs_std:.2f}')

            print(f'Info: {info}, frac: {frac}, mf1: {result}')

    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-{args.dataset}-agg_perfs.csv', index=False)
    


def get_cb_han_perfs():

    args.dataset = 'craigslistbargain'

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']

    results_dir = f'../baselines/results/{args.dataset}/'
    infos       = ['None_None', 'intentions_None', 'None_traditional_summary', 'None_scd_summary', 'None_scm_summary', 'intentions_scd_summary', 'intentions_scm_summary', 'intentions_traditional_summary']

    mapping_dict = {
        'None_None': 'UTT',
        'None_traditional_summary': 'UTT + Trad',
        'None_scd_summary': 'UTT + SCD',
        'None_scm_summary': 'UTT + SCM',
        'intentions_None': 'UTT+ INT',
        'intentions_scd_summary': 'UTT+ INT + SCD',
        'intentions_scm_summary': 'UTT + INT + SCM',
        'intentions_traditional_summary': 'UTT + INT + Trad'        
    }

    results_dict = ddict(list)

    for info in infos:

        results_dict['Info'].append(mapping_dict[info])
        results_dict['Measure'].append('MF1')

        results_dict['Info'].append(mapping_dict[info])
        results_dict['Measure'].append('RMSE Score')

        for frac in fracs:

            mf1_arr     = []
            score_arr   = []

            for seed in seeds:
                
                buyer_ytrue     = []
                buyer_ypred     = []
                curr_score_arr  = []

                for fold in range(0, 5):

                    csv_file = f'{results_dir}/seed_{seed}/{args.dataset}_regression_ratio_{frac}_{info}_bert-base-uncased_fold_{fold}_test.csv'

                    try:
                        df       = pd.read_csv(csv_file)
                        # dialogue_id,utterance,gold_score,predicted_score,buyer_target,seller_target,sale_price

                        buyer_targets       = df['buyer_target'].values
                        seller_targets      = df['seller_target'].values
                        sale_prices         = df['sale_price'].values
                        predicted_scores    = df['predicted_score'].values
                        gold_scores         = df['gold_score'].values

                        for bt, st, sp, ps, gs in zip(buyer_targets, seller_targets, sale_prices, predicted_scores, gold_scores):

                            true_score = (sp - bt)/(st - bt)

                            # assert true_score == gs

                            pred_price = ps * (st - bt) + bt

                            if 2* pred_price < st + bt:
                                pred_deal = 1
                            else:
                                pred_deal = 0
                            
                            if 2* sp < st + bt:
                                true_deal = 1
                            else:
                                true_deal = 0
                            
                            buyer_ytrue.append(true_deal)
                            buyer_ypred.append(pred_deal)

                            curr_score_arr.append((gs - ps)**2)

                    except Exception as e:
                        print(f'Error loading {csv_file}: {e}')
                        continue

                # dialogue_id,fold,text,label,predicted_label,correct,confidence_score

                mf1     = f1_score(buyer_ytrue, buyer_ypred, average='macro')
                mf1_arr.append(100*mf1)

                err_score = np.mean(curr_score_arr)
                score_arr.append(100*err_score)
            
            # calculate the mean and std of the mf1
            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}'             
            results_dict[frac].append(result)

            score_mean = np.mean(score_arr)
            score_std  = np.std(score_arr)

            results_dict[frac].append(f'{score_mean:.2f} ± {score_std:.2f}')

            print(f'Info: {info}, frac: {frac}, mf1: {result}')

    # convert the results_dict to a dataframe
    results_df = pd.DataFrame(results_dict)
    
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/{args.method}-cb-agg_perfs.csv', index=False)
    
        
        
### Get the SFT performances ####


def get_p4g_sft_perfs():

    fracs       = [0.25, 0.375, 0.5, 0.625, 0.75]

    seeds       = ['42', '11611', '10623']


    results_dir = f'../src/{args.method}/results/{args.dataset}/'
    infos       = ['none', 'local', 'global_scd', 'global_scm', 'global_traditional', 'both_scd', 'both_scm', 'both_traditional']
    
    results_dict = ddict(list)

    for info in infos:

        results_dict['Info'].append(info)
        for frac in fracs:

            mf1_arr = []
            for seed in seeds:

                
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
    



def get_cd_sft_perfs():

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


### ICL PERFS  ###

## P4G Dataset ##
def get_p4g_icl_perfs():

    results_dir =f'/data/shire/projects/RAT_forecast/FOLIAGE/src/icl/results/{args.dataset}/llama70b/'

    p4g_results_dict = ddict(list)

    for seed in os.listdir(results_dir):

        scaffoldings = ['baseline', 'dualscaffolding', 'localscaffolding', 'globalscaffolding']

        local_infos     = ['no_intentions']
        global_infos    = ['none']

        for scaffolding in scaffoldings:
            
            if 'dual' in scaffolding:
                global_infos = ['scd', 'scm', 'traditional']
                local_infos  = ['with_intentions']
            
            if 'global' in scaffolding:
                global_infos = ['scd', 'scm', 'traditional']
                local_infos  = ['no_intentions']
            
            if 'local' in scaffolding:
                global_infos = ['none']
                local_infos  = ['with_intentions']

            splits = [0.25, 0.375, 0.5, 0.625, 0.75]

            for local_info in local_infos:
                for global_info in global_infos:

                    for split in splits:
                        
                        csv_file    = f'{results_dir}/{seed}/{scaffolding}/{args.dataset}_{split}_{global_info}_{local_info}_llama70b_final_results.csv'

                        df          = pd.read_csv(csv_file)

                        for idx, row in df.iterrows():
                            
                            p4g_results_dict['seed'].append(seed)
                            p4g_results_dict['scaffold'].append(f'{global_info}_{local_info}')
                            p4g_results_dict['frac'].append(split)
                            p4g_results_dict['dialogue_id'].append(row['dialogue_id'])

                            p4g_results_dict['gold_label'].append(1 if row['actual'] =='yes' else 0)
                            p4g_results_dict['predicted_label'].append(1 if row['predicted']=='yes' else 0)
                        

                        print(f'Done for {split} {local_info} {global_info} {scaffolding}')
                            
    # convert the results_dict to a dataframe   
    results_df = pd.DataFrame(p4g_results_dict)            
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/icl-{args.dataset}-all_perfs.csv', index=False)


    final_results = ddict(list)

    for scaffold in results_df['scaffold'].unique():
        
        for seed in results_df['seed'].unique():

            for frac in results_df['frac'].unique():

                df_subset = results_df[(results_df['scaffold'] == scaffold) & (results_df['frac'] == frac) & (results_df['seed'] == seed)]

                y_trues = df_subset['gold_label'].values
                y_preds = df_subset['predicted_label'].values

                mf1     = f1_score(y_trues, y_preds, average='macro')

                final_results['seed'].append(seed)
                final_results['scaffold'].append(scaffold)
                final_results['frac'].append(frac)
                final_results['mf1'].append(100*mf1)
            
            df_subset = results_df[(results_df['scaffold'] == scaffold) & (results_df['seed'] == seed)]

            y_trues = df_subset['gold_label'].values
            y_preds = df_subset['predicted_label'].values

            mf1     = f1_score(y_trues, y_preds, average='macro')

            final_results['seed'].append(seed)
            final_results['scaffold'].append(scaffold)
            final_results['frac'].append('All')
            final_results['mf1'].append(100*mf1)
    
    final_results = pd.DataFrame(final_results)

    agg_perf = ddict(list)

    mapping_dict = {
        'none_no_intentions': 'UTT',
        'traditional_no_intentions': 'UTT + Trad',
        'scd_no_intentions': 'UTT + SCD',
        'scm_no_intentions': 'UTT + SCM',
        'none_with_intentions': 'UTT+ INT',
        'scd_with_intentions': 'UTT+ INT + SCD',
        'scm_with_intentions': 'UTT + INT + SCM',
        'traditional_with_intentions': 'UTT + INT + Trad'        
    }
   

    for scaffold in final_results['scaffold'].unique():
        
        agg_perf['Info'].append(mapping_dict[scaffold])

        for frac in final_results['frac'].unique():

            df_subset = final_results[(final_results['scaffold'] == scaffold) & (final_results['frac'] == frac)]
            mf1_arr = df_subset['mf1'].values

            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}' 
            
            
            agg_perf[frac].append(result)

            # agg_perf[frac].append(result)
            print(f'Scaffold: {scaffold}, frac: {frac}, mf1: {result}')

    agg_perf_df = pd.DataFrame(agg_perf)
    agg_perf_df.to_csv(f'../results/icl-{args.dataset}-agg_perfs.csv', index=False)
        
    


### CB Dataset ###
def get_cb_icl_perfs():

    results_dir =f'/data/shire/projects/RAT_forecast/FOLIAGE/src/icl/results/{args.dataset}/llama70b/'

    eps = 1e-10
    results_dict = ddict(list)
    err_cases  = 0

    for seed in os.listdir(results_dir):

        scaffoldings = ['baseline', 'dualscaffolding', 'localscaffolding', 'globalscaffolding']

        local_infos     = ['no_intentions']
        global_infos    = ['none']

        for scaffolding in scaffoldings:
            
            if 'dual' in scaffolding:
                global_infos = ['scd', 'scm', 'traditional']
                local_infos  = ['with_intentions']
            
            if 'global' in scaffolding:
                global_infos = ['scd', 'scm', 'traditional']
                local_infos  = ['no_intentions']
            
            if 'local' in scaffolding:
                global_infos = ['none']
                local_infos  = ['with_intentions']

            splits = [0.25, 0.375, 0.5, 0.625, 0.75]

            for local_info in local_infos:
                for global_info in global_infos:

                    for split in splits:
                        
                        csv_file    = f'{results_dir}/{seed}/{scaffolding}/{args.dataset}_{split}_{global_info}_{local_info}_llama70b_final_results.csv'

                        df          = pd.read_csv(csv_file)

                        for idx, row in df.iterrows():

                            #,predicted_final_price,buyer_target,seller_target,sale_price,price_error,percent_error

                            true_score = (row['sale_price'] - row['buyer_target'])/(row['seller_target'] - row['buyer_target'] + eps)
                            pred_score = (row['predicted_final_price'] - row['buyer_target'])/(row['seller_target'] - row['buyer_target']+ eps)

                            if true_score != true_score:
                                err_cases += 1
                                continue
                            
                            if pred_score != pred_score:
                                err_cases += 1
                                continue
                            
                            results_dict['seed'].append(seed)
                            results_dict['scaffold'].append(f'{global_info}_{local_info}')
                            results_dict['frac'].append(split)
                            results_dict['dialogue_id'].append(row['dialogue_id'])

                            results_dict['buyer_target'].append(row['buyer_target'])
                            results_dict['seller_target'].append(row['seller_target'])
                            results_dict['sale_price'].append(row['sale_price'])
                            results_dict['predicted_final_price'].append(row['predicted_final_price'])
                            
                            diff_price      = row['sale_price'] - row['predicted_final_price']
                            norm_diff_price = diff_price /(row['sale_price'] + eps)

                            results_dict['true_score'].append(true_score)
                            results_dict['pred_score'].append(pred_score)
                            results_dict['diff_price'].append(diff_price)
                            results_dict['norm_diff_price'].append(norm_diff_price)


                            if 2* row['sale_price'] < row['seller_target'] + row['buyer_target']:
                                results_dict['true_deal'].append(1)
                            else:
                                results_dict['true_deal'].append(0)
                            
                            if 2* row['predicted_final_price'] < row['seller_target'] + row['buyer_target']:
                                results_dict['pred_deal'].append(1)
                            else:
                                results_dict['pred_deal'].append(0)



                        print(f'Done for {split} {local_info} {global_info} {scaffolding}')

    print(f'Total error cases due to NaN scores: {err_cases}')                            
    # convert the results_dict to a dataframe   
    results_df = pd.DataFrame(results_dict)            
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/icl-{args.dataset}-all_perfs.csv', index=False)

    final_results = ddict(list)

    for scaffold in results_df['scaffold'].unique():
        
        for seed in results_df['seed'].unique():

            for frac in results_df['frac'].unique():

                df_subset = results_df[(results_df['scaffold'] == scaffold) & (results_df['frac'] == frac) & (results_df['seed'] == seed)]

                y_trues = df_subset['true_score'].values
                y_preds = df_subset['pred_score'].values
                true_deals = df_subset['true_deal'].values
                pred_deals = df_subset['pred_deal'].values

                
                succ_corr = pearsonr(y_trues, y_preds)[0]
                rmse      = np.sqrt(np.mean((y_trues - y_preds)**2))
            
                rmse_norm_diff_prices = np.sqrt(np.mean((df_subset['norm_diff_price'].values)**2))
                # df_subset['norm_diff_price'] = np.abs(df_subset['norm_diff_price'].values)

                # y_trues = df_subset['gold_label'].values
                # y_preds = df_subset['predicted_label'].values

                deal_mf1     = f1_score(true_deals, pred_deals, average='macro')
                deal_kappa   = cohen_kappa_score(true_deals, pred_deals)

                final_results['seed'].append(seed)
                final_results['scaffold'].append(scaffold)
                final_results['frac'].append(frac)
                final_results['corr'].append(succ_corr)
                final_results['rmse'].append(rmse)
                final_results['rmse_norm_diff_price'].append(rmse_norm_diff_prices)
                final_results['deal_mf1'].append(100*deal_mf1)
                final_results['deal_kappa'].append(deal_kappa)

            df_subset = results_df[(results_df['scaffold'] == scaffold) & (results_df['seed'] == seed)]

            y_trues = df_subset['true_score'].values
            y_preds = df_subset['pred_score'].values
            true_deals = df_subset['true_deal'].values
            pred_deals = df_subset['pred_deal'].values

            succ_corr = pearsonr(y_trues, y_preds)[0]
            rmse      = np.sqrt(np.mean((y_trues - y_preds)**2))

            rmse_norm_diff_prices = np.sqrt(np.mean((df_subset['norm_diff_price'].values)**2))

            deal_mf1     = f1_score(true_deals, pred_deals, average='macro')
            deal_kappa   = cohen_kappa_score(true_deals, pred_deals)
            
            final_results['seed'].append(seed)
            final_results['scaffold'].append(scaffold)
            final_results['frac'].append('All')
            final_results['corr'].append(succ_corr)
            final_results['rmse'].append(rmse)
            final_results['rmse_norm_diff_price'].append(rmse_norm_diff_prices)
            final_results['deal_mf1'].append(100*deal_mf1)
            final_results['deal_kappa'].append(deal_kappa)
    
    final_results = pd.DataFrame(final_results)

    agg_perf = ddict(list)


    mapping_dict = {
        'none_no_intentions': 'UTT',
        'traditional_no_intentions': 'UTT + Trad',
        'scd_no_intentions': 'UTT + SCD',
        'scm_no_intentions': 'UTT + SCM',
        'none_with_intentions': 'UTT+ INT',
        'scd_with_intentions': 'UTT+ INT + SCD',
        'scm_with_intentions': 'UTT + INT + SCM',
        'traditional_with_intentions': 'UTT + INT + Trad'        
    }

    for scaffold in final_results['scaffold'].unique():
        
        agg_perf['Info'].append(mapping_dict[scaffold])

        for frac in final_results['frac'].unique():

            df_subset = final_results[(final_results['scaffold'] == scaffold) & (final_results['frac'] == frac)]

            # corr_arr  = df_subset['corr'].values
            # rmse_arr  = df_subset['rmse'].values

            # corr_mean = np.mean(corr_arr)
            # corr_std  = np.std(corr_arr)

            # result  = f'{corr_mean:.2f} ± {corr_std:.2f}' 
            
            # agg_perf[f'{frac}_corr'].append(result)

            # # print(f'Scaffold: {scaffold}, frac: {frac} corr_result: {result}')

            # rmse_mean = np.mean(rmse_arr)
            # rmse_std  = np.std(rmse_arr)

            # result  = f'{rmse_mean:.2f} ± {rmse_std:.2f}'
            # agg_perf[f'{frac}_rmse'].append(result)

            # # agg_perf[frac].append(result)
            # # print(f'Scaffold: {scaffold}, frac: {frac} rmse_result: {result}')

            # rmse_norm_diff_price_arr  = df_subset['rmse_norm_diff_price'].values
            # rmse_norm_diff_price_mean = np.mean(rmse_norm_diff_price_arr)
            # rmse_norm_diff_price_std  = np.std(rmse_norm_diff_price_arr)
            # result  = f'{rmse_norm_diff_price_mean:.2f} ± {rmse_norm_diff_price_std:.2f}'

            # agg_perf[f'{frac}_rmse_norm_diff_price'].append(result)

            deal_mf1_arr = df_subset['deal_mf1'].values
            deal_mf1_mean = np.mean(deal_mf1_arr)
            deal_mf1_std  = np.std(deal_mf1_arr)
            result  = f'{deal_mf1_mean:.2f} ± {deal_mf1_std:.2f}'

            agg_perf[f'{frac}'].append(result)

            # deal_kappa_arr = df_subset['deal_kappa'].values
            # deal_kappa_mean = np.mean(deal_kappa_arr)
            # deal_kappa_std  = np.std(deal_kappa_arr)
            # result  = f'{deal_kappa_mean:.2f} ± {deal_kappa_std:.2f}'

            # agg_perf[f'{frac}-kappa'].append(results)            

    agg_perf_df = pd.DataFrame(agg_perf)
    agg_perf_df.to_csv(f'../results/icl-{args.dataset}-agg_perfs.csv', index=False)


## Conversational Derailment Dataset ##
def get_cd_icl_perfs():

    results_dir =f'/data/shire/projects/RAT_forecast/FOLIAGE/src/icl/results/{args.dataset}/llama70b/'

    results_dict = ddict(list)

    for seed in os.listdir(results_dir):

        scaffoldings = ['baseline', 'dualscaffolding', 'localscaffolding', 'globalscaffolding']

        local_infos     = ['no_intentions']
        global_infos    = ['none']

        for scaffolding in scaffoldings:
            
            if 'dual' in scaffolding:
                global_infos = ['scd', 'scm', 'traditional']
                local_infos  = ['with_intentions']
            
            if 'global' in scaffolding:
                global_infos = ['scd', 'scm', 'traditional']
                local_infos  = ['no_intentions']
            
            if 'local' in scaffolding:
                global_infos = ['none']
                local_infos  = ['with_intentions']

            splits = [0.25, 0.375, 0.5, 0.625, 0.75]

            for local_info in local_infos:
                for global_info in global_infos:

                    for split in splits:
                        
                        csv_file    = f'{results_dir}/{seed}/{scaffolding}/{args.dataset}_{split}_{global_info}_{local_info}_llama70b_final_results.csv'

                        df          = pd.read_csv(csv_file)

                        for idx, row in df.iterrows():
                            results_dict['seed'].append(seed)
                            results_dict['scaffold'].append(f'{global_info}_{local_info}')
                            results_dict['frac'].append(split)
                            results_dict['dialogue_id'].append(row['dialogue_id'])

                            results_dict['gold_label'].append(1 if row['actual'] =='yes' else 0)
                            results_dict['predicted_label'].append(1 if row['predicted']=='yes' else 0)
                        

                        print(f'Done for {split} {local_info} {global_info} {scaffolding}')
                            
    # convert the results_dict to a dataframe   
    results_df = pd.DataFrame(results_dict)            
    # save the dataframe to a csv file
    results_df.to_csv(f'../results/icl-{args.dataset}-all_perfs.csv', index=False)



    final_results = ddict(list)

    for scaffold in results_df['scaffold'].unique():
        
        for seed in results_df['seed'].unique():

            for frac in results_df['frac'].unique():

                df_subset = results_df[(results_df['scaffold'] == scaffold) & (results_df['frac'] == frac) & (results_df['seed'] == seed)]

                y_trues = df_subset['gold_label'].values
                y_preds = df_subset['predicted_label'].values

                mf1     = f1_score(y_trues, y_preds, average='macro')

                final_results['seed'].append(seed)
                final_results['scaffold'].append(scaffold)
                final_results['frac'].append(frac)
                final_results['mf1'].append(100*mf1)
            
            df_subset = results_df[(results_df['scaffold'] == scaffold) & (results_df['seed'] == seed)]

            y_trues = df_subset['gold_label'].values
            y_preds = df_subset['predicted_label'].values

            mf1     = f1_score(y_trues, y_preds, average='macro')

            final_results['seed'].append(seed)
            final_results['scaffold'].append(scaffold)
            final_results['frac'].append('All')
            final_results['mf1'].append(100*mf1)
    
    final_results = pd.DataFrame(final_results)

    agg_perf = ddict(list)


    mapping_dict = {
        'none_no_intentions': 'UTT',
        'traditional_no_intentions': 'UTT + Trad',
        'scd_no_intentions': 'UTT + SCD',
        'scm_no_intentions': 'UTT + SCM',
        'none_with_intentions': 'UTT+ INT',
        'scd_with_intentions': 'UTT+ INT + SCD',
        'scm_with_intentions': 'UTT + INT + SCM',
        'traditional_with_intentions': 'UTT + INT + Trad'        
    }

    for scaffold in final_results['scaffold'].unique():
        
        agg_perf['Info'].append(mapping_dict[scaffold])

        for frac in final_results['frac'].unique():

            df_subset = final_results[(final_results['scaffold'] == scaffold) & (final_results['frac'] == frac)]
            mf1_arr = df_subset['mf1'].values

            mf1_mean = np.mean(mf1_arr)
            mf1_std  = np.std(mf1_arr)

            result  = f'{mf1_mean:.2f} ± {mf1_std:.2f}' 
            
            
            agg_perf[frac].append(result)

            # agg_perf[frac].append(result)
            print(f'Scaffold: {scaffold}, frac: {frac}, mf1: {result}')

    agg_perf_df = pd.DataFrame(agg_perf)
    agg_perf_df.to_csv(f'../results/icl-{args.dataset}-agg_perfs.csv', index=False)
        
    








if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='p4g')
    parser.add_argument('--step',    type=str, required=True)
    parser.add_argument('--method',  type=str, default='sft')
    
    
    args = parser.parse_args()


    if args.step == 'get_perf':

        if args.method == 'sft':
    
            if args.dataset == 'p4g':
                get_p4g_sft_perfs()
            
            elif args.dataset == 'cd':
                get_cd_sft_perfs()

            elif args.dataset == 'casino':
                get_casino_perfs()
            
            elif args.dataset == 'cd':
                get_cd_perfs()
        

        if args.method == 'han':
    
            if args.dataset == 'p4g':
                get_p4g_han_perfs()
            
            elif args.dataset == 'cd':
                get_cd_han_perfs()
            
            elif args.dataset == 'cb':
                get_cb_han_perfs()
        

        if args.method == 'icl':
            if args.dataset == 'p4g':
                get_p4g_icl_perfs()
            
            elif args.dataset == 'cb':
                get_cb_icl_perfs()
            
            elif args.dataset == 'cd':
                get_cd_icl_perfs()