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

def preprocess_CB(args):

    #dialogue_id,buyer_target,seller_target,sale_price,utterance_idx,speaker,utterance,intention,scm_summary,scd_summary,politeness_theory_stage2_summary,traditional_summary,relational_summary

    fracwise_dialogs    = ddict(list)

    df = pd.read_csv(f'../data/{args.dataset}/utterance_level_data_ratio_{args.frac}.csv')
    df.rename(columns={'politeness_theory_stage2_summary': 'politeness_summary'}, inplace=True)
    uniq_conversations  = df['dialogue_id'].unique()

    foldwise_conv_ids   = split_ids2folds(uniq_conversations, args.num_folds)

    test_foldwise_data   = ddict(list)
    train_foldwise_data  = ddict(list)

    for frac in [0.25, 0.375, 0.5, 0.625, 0.75]:

        df = pd.read_csv(f'../data/{args.dataset}/utterance_level_data_ratio_{frac}.csv')
        df.rename(columns={'politeness_theory_stage2_summary': 'politeness_summary'}, inplace=True)
        uniq_conversations  = df['dialogue_id'].unique()

        
        for fold in range(args.num_folds):

            test_conv_data  = []
            train_conv_data = []

            for dialog_id in uniq_conversations:
                
                curr_data       = {'dialogue_id': dialog_id, \
                        'speakers': [],\
                        'utterances': [],\
                        'intentions': [],\
                        'buyer_target':None,\
                        'seller_target':None,\
                        'sale_price':None, \
                        'traditional_summary':"", \
                        'relational_summary':"", \
                        'scm_summary':"", \
                        'scd_summary':"", \
                        'politeness_summary':""}

                curr_df = df[df['dialogue_id'] == dialog_id]
                dialog_history = []

                for idx, row in curr_df.iterrows():
                    curr_data['speakers'].append(row['speaker'])
                    curr_data['utterances'].append(row['utterance'])
                    curr_data['intentions'].append(row['intention'])

                    if curr_data['buyer_target'] == None:
                        curr_data['buyer_target'] = row['buyer_target']
                    else:                    
                        assert row['buyer_target'] == curr_data['buyer_target']

                    
                    if curr_data['seller_target'] == None:
                        curr_data['seller_target'] = row['seller_target']
                    else:
                        assert row['seller_target'] == curr_data['seller_target']
                    
                    if curr_data['sale_price'] == None:
                        curr_data['sale_price'] = row['sale_price']
                    else:
                        assert row['sale_price'] == curr_data['sale_price']

                    if curr_data['traditional_summary'] == "":
                        curr_data['traditional_summary'] = row['traditional_summary']
                    else:
                        assert row['traditional_summary'] == curr_data['traditional_summary']
                    
                    if curr_data['relational_summary'] == "":
                        curr_data['relational_summary'] = row['relational_summary']
                    else:
                        assert row['relational_summary'] == curr_data['relational_summary']
                    
                    if curr_data['scm_summary'] == "":
                        curr_data['scm_summary'] = row['scm_summary']
                    else:
                        assert row['scm_summary'] == curr_data['scm_summary']
                    
                    if curr_data['scd_summary'] == "":
                        curr_data['scd_summary'] = row['scd_summary']
                    else:
                        assert row['scd_summary'] == curr_data['scd_summary']
                    
                    if curr_data['politeness_summary'] == "":
                        curr_data['politeness_summary'] = row['politeness_summary']
                    else:
                        assert row['politeness_summary'] == curr_data['politeness_summary']



                BT = curr_data['buyer_target']
                ST = curr_data['seller_target']
                SP = curr_data['sale_price']

                score = 0

                score = (SP- BT)/ (ST -BT)

                try:
                    assert score >=0 and score <= 1
                except:
                    import pdb; pdb.set_trace()

                if score >0.5:
                    curr_data['binary_score']  = 1
                else:
                    curr_data['binary_score']  = 0

                curr_data['score'] = score

                if dialog_id in foldwise_conv_ids[fold]:
                    test_conv_data.append(curr_data)
                    test_foldwise_data[fold].append(curr_data)

                else:
                    train_conv_data.append(curr_data)
                    train_foldwise_data[fold].append(curr_data)
            
            

            with open(f'../data/{args.dataset}/processed/RAT_{frac}_{fold}_train.json', 'w') as f:
                json.dump(train_conv_data, f, indent=4)
            
            with open(f'../data/{args.dataset}/processed/RAT_{frac}_{fold}_test.json', 'w') as f:
                json.dump(test_conv_data, f, indent=4)
        
            # print("Length of train data: ", len(train_conv_data))
            # print("Length of test data: ", len(test_conv_data))
    
    for fold in range(args.num_folds):

        with open(f'../data/{args.dataset}/processed/RAT_ALL_{fold}_train.json', 'w') as f:
            json.dump(train_foldwise_data[fold], f, indent=4)
        
        with open(f'../data/{args.dataset}/processed/RAT_ALL_{fold}_test.json', 'w') as f:
            json.dump(test_foldwise_data[fold], f, indent=4)
    
        median_train_score = np.median([x['score'] for x in train_foldwise_data[fold]])
        median_test_score = np.median([x['score'] for x in test_foldwise_data[fold]])

        print(f"Fold {fold}: Train score: {median_train_score}, Test score: {median_test_score}")



def preprocess_wiki(args):

    df = pd.read_csv('../data/wiki/wiki.csv')
    invalid_dialog_ids_list = json.load(open(f'../data/{args.dataset}/invalid_dialog_ids.json'))

    columns = df.columns.tolist()
    print(columns)

    # 'utterance_idx', 'speaker', 'conversation_id', 'text', 'comment_has_personal_attack', 'is_section_header'

    print("Number of utterances: ", len(df))
    print("Number of unique conversations: ", len(df['conversation_id'].unique()))

    df  = df[df['is_section_header'] == False]

    uniq_conversations = df['conversation_id'].unique()

    utt_conv_length = []

    for conv_id in uniq_conversations:
        curr_df = df[df['conversation_id'] == conv_id]
        utt_conv_length.append(len(curr_df))
    
    print("Average number of utterances per conversation: ", np.mean(utt_conv_length))

    print(Counter(utt_conv_length))

    loc_per_attack = []

    for conv_id in uniq_conversations:

        curr_df = df[df['conversation_id'] == conv_id]

        utt_idx = 0

        for idx, row in curr_df.iterrows():
            utt_idx += 1
            if row['comment_has_personal_attack'] == True:
                loc_per_attack.append(utt_idx/len(curr_df))                

    print("Average location of personal attack: ", np.mean(loc_per_attack))

    #### A mean value of 1.0 implies that the attacks are always at the end of the conversation ####

    print(df['comment_has_personal_attack'].value_counts())

    wiki_data = []

    null_texts = 0

    for conv_id in uniq_conversations:

        curr_df = df[df['conversation_id'] == conv_id]

        speakers       = list(curr_df['speaker'].unique())

        speaker_map    = {}

        for speaker_idx, speaker in enumerate(speakers):
            speaker_map[speaker] = f'S_{speaker_idx}'


        dialog_history = []

        for idx, row in curr_df.iterrows():

            curr_data = {}

            assert row['conversation_id'] == conv_id and row['is_section_header'] == False

            '''
            "dialog_id": "train_0",
            "turn_id": "2",
            "speaker": "ER",
            "utterance": "The research team is helping an organization called 'Save the Children'.",
            "face_act": "other",
            "dialog_history": [
                [
                    "ER",
                    "Hi..."
                ],
                [
                    "ER",
                    "how are you today?"
                ]
            ]
            '''

            if row['text'].strip() == "":
                null_texts += 1
                invalid_dialog_ids_list.append(row['conversation_id'])
                continue

            curr_data['dialog_id']      = row['conversation_id']
            curr_data['turn_id']        = row['utterance_idx']
            curr_data['speaker']        = speaker_map[row['speaker']]
            curr_data['utterance']      = row['text']
            curr_data['personal_attack']= 1 if row['comment_has_personal_attack'] == True else 0
            curr_data['dialog_history'] = list(dialog_history)

            wiki_data.append(curr_data)


            dialog_history.append([speaker_map[row['speaker']], row['text']])


    with open(f'../data/{args.dataset}/{args.dataset}-train.json', 'w') as f:
        json.dump(wiki_data, f, indent=4)

    print("Number of null texts: ", null_texts)

    invalid_dialog_ids_list = list(set(invalid_dialog_ids_list))

    with open(f'../data/{args.dataset}/invalid_dialog_ids.json', 'w') as f:
        json.dump(invalid_dialog_ids_list, f, indent=4)


def incorporate_wiki(args):


    data     = json.load(open(f'../data/{args.dataset}/{args.dataset}-train.json'))

    all_data = pd.read_csv(f'../data/{args.dataset}/{args.dataset}-gpt-4o_speaker_intention_data.csv')

    

    RA_augmented_data = []
    incorrect_rows    = 0

    for idx, elem in enumerate(data):
        curr_data = elem

        #     {
        #     "dialog_id": "143890867.11926.11926",
        #     "turn_id": "144065917.12226.12226",
        #     "speaker": "S_1",
        #     "utterance": "I assume your deliberate lying has a point, but get over it.  Stop bizarrely goin on about transwiking.  That has nothing to do with the afd.  There was a plain consensus for deleting the article. Transwikiing is completely unrelated. Please don't be so deliberately obtuse in the future.  wasting other people's time is simply rude. ",
        #     "personal_attack": 1,
        #     "dialog_history": [
        #         [
        #             "S_0",
        #             "No more than two editors advocated deletion.  Greatestrowerever and maybe Kymacpherson.  That's not a clear consensus for deletion.  Cheers, "
        #         ],
        #         [
        #             "S_1",
        #             "In the future please don't close Afds when you don't have the courtesy of reading the comments.  All comments favored deletion except two.  Please don't be so careless in the future. "
        #         ],
        #         [
        #             "S_0",
        #             "That simply isn't true.  If you read the comments, you'll find it's actually 2 keep, 4 transwiki, 2 delete (more or less).  The ''comments'' favour no consensus/transwiki.  The \"votes\" favour delete, but voting is evil, of course... "
        #         ],
        #         [
        #             "S_0",
        #             "Somehow, I suspect you may wish to participate in wiki_link discussion.  Cheers, "
        #         ]
        #     ]
        # },
        
        dialog_id = elem['dialog_id']
        turn_id = elem['turn_id']
        speaker = elem['speaker']
        utterance = elem['utterance'].strip()
        dialog_history = elem['dialog_history']

        # dataset,dialog_id,utterance_idx,speaker,utterance,gpt-4o_speaker_intention

        corr_row = all_data[(all_data['dialog_id'] == dialog_id) & (all_data['utterance_idx'] == int(len(dialog_history))) & (all_data['speaker'] == speaker)] #& (all_data['utterance']==utterance)]

        try:
            assert len(corr_row) == 1

            utt2 = corr_row['utterance'].values[0].strip()

            assert utt2.strip() == utterance.strip()

        except:
            # import pdb; pdb.set_trace()
    
            incorrect_rows += 1
            print(dialog_id, turn_id, speaker, utterance, utt2)

            continue
            
        curr_data['gpt-4o_speaker_intention'] = corr_row['gpt-4o_speaker_intention'].values[0]

        RA_augmented_data.append(curr_data)

    print(f'Length of RA augmented data: {len(RA_augmented_data)}')
    print(f'Length of original data: {len(data)}')

    # print(f'Incorrect rows for: {incorrect_rows}/{len(split_data)}')

    with open(f'../data/{args.dataset}/RA-{args.dataset}.json', 'w') as f:
        json.dump(RA_augmented_data, f, indent=4)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--dataset', type=str, default='CB', help='Path to the data directory.')
    parser.add_argument('--step',    type=str, default='prepare', required=True)
    parser.add_argument('--frac',    type=str, default=0.25, help='Fraction of data to use.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation.')
    
    args = parser.parse_args()

    if args.step == 'prepare':

        if args.dataset == 'CB':
            preprocess_CB(args)

        if args.dataset == 'wiki':
            preprocess_wiki(args)

    if args.step == 'add':
        if args.dataset == 'wiki':
            incorporate_wiki(args)




