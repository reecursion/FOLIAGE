from collections import defaultdict
import random
from typing import Union
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
import numpy as np
import argparse
from transformers import AutoTokenizer


def create_mini_batch(samples):
    utt_ids         = [s["utt_ids"] for s in samples]
    utt_mask        = [s["utt_masks"] for s in samples]
    global_ids      = [s["global_ids"] for s in samples]
    global_mask     = [s["global_masks"] for s in samples]
    binary_score    = [s["binary_score"] for s in samples]
    utterances      = [s["utterances"] for s in samples]
    speakers        = [s["speakers"] for s in samples]
    dialogue_id     = [s["dialogue_id"] for s in samples]

    if any(g_id is None for g_id in global_ids):
        global_ids_tensor = None
        global_mask_tensor = None
    else:
        global_ids_tensor = torch.vstack(global_ids)
        global_mask_tensor = torch.vstack(global_mask)

    return {
        "utt_ids":  utt_ids,
        "utt_masks": utt_mask,
        "global_ids": global_ids_tensor,
        "global_masks": global_mask_tensor,
        "binary_score": torch.LongTensor(binary_score),
        "utt_text": utterances,
        "speakers": speakers,
        "dialogue_id": dialogue_id,
    }



def conv_encoder(item, tokenizer, local_info=None, global_info=None, num_classes=2):
    #     "dialogue_id": "C_4566b2e6938a4a6a8e714e799d780e71",
    #     "speakers": [
    #         "Buyer",
    #         "Seller"
    #     ],
    #     "utterances": [
    #         "Hi,I am interested in your board!",
    #         "Hi, do you have any questions?"
    #     ],
    #     "intentions": [
    #         "Expressing interest in the product",
    #         "Offering assistance and inviting questions"
    #     ],
    #     "buyer_target": 120,
    #     "seller_target": 200,
    #     "sale_price": 145.0,
    #     "traditional_summary": "The buyer expresses interest in purchasing a board, and the seller responds by asking if the buyer has any questions.",
    #     "relational_summary": "The buyer seems interested and open, indicating a willingness to engage further. The seller appears approachable and ready to assist, suggesting a professional and accommodating attitude. There is a neutral level of respect and trust, as the interaction is just beginning. Both parties seem aligned in their purpose, with no signs of frustration. The rapport is minimal but positive, as they are both engaging politely and constructively.",
    #     "scm_summary": "Buyer:  \n  warmth: high  \n  competence: high  \n  explanation: The Buyer initiates the conversation with a friendly greeting and expresses interest, indicating openness and positive intent. Their direct approach suggests they are knowledgeable about what they want.\n\nSeller:  \n  warmth: high  \n  competence: high  \n  explanation: The Seller responds promptly with a friendly greeting and offers assistance, demonstrating a willingness to help and engage. Their readiness to answer questions suggests they are knowledgeable and prepared.",
    #     "scd_summary": "The conversation begins with the buyer expressing interest in the seller's product, indicating a positive and curious sentiment. The seller responds promptly and openly, inviting further engagement by asking if the buyer has any questions. This sets a cooperative tone, with the seller adopting a supportive and accommodating strategy to facilitate the buyer's decision-making process. Both parties appear open and ready to continue the dialogue constructively.",
    #     "politeness_summary": "In this exchange, both participants effectively use face management strategies to maintain a positive social dynamic. The Buyer raises the Seller's positive face by expressing interest, affirming the product's value. The Seller's response invites further engagement, respecting the Buyer's autonomy and minimizing face-threatening acts by allowing the Buyer to guide the conversation. These politeness strategies foster a respectful and cooperative interaction, reinforcing positive interpersonal dynamics and mutual respect, which can lead to a successful transaction.",
    #     "binary_score": 0,

    utt_ids     = []
    utt_mask    = []

    global_ids  = None
    global_mask = None

    if local_info is not None:
        local_infos = item[local_info]
    else:
        local_infos = [None] * len(item['speakers'])

    for speaker, utterance, curr_info in zip(item['speakers'], item['utterances'], local_infos):

        if curr_info is not None:
            utt_data = f'{speaker}{tokenizer.sep_token}{utterance}{tokenizer.sep_token}{local_info}'
        else:
            utt_data = f'{speaker}{tokenizer.sep_token}{utterance}'
        
        dial_encoding                     = tokenizer.encode_plus(
                    utt_data,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
        utt_ids.append(dial_encoding['input_ids'])
        utt_mask.append(dial_encoding['attention_mask'])
    

    utt_ids     = torch.cat(utt_ids, dim=0)
    utt_mask    = torch.cat(utt_mask, dim=0)

    if global_info is not None:
        global_summary = item[global_info]
        global_encoding                     = tokenizer.encode_plus(
                    global_summary,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
        global_ids = global_encoding['input_ids']
        global_mask = global_encoding['attention_mask']
    
    
    binary_score    = item['binary_score']


    result = {
        'utt_ids': utt_ids,
        'utt_masks': utt_mask,
        'global_ids': global_ids,
        'global_masks': global_mask,
        'binary_score': binary_score,
    }

    return result

class DialogueDataset(Dataset):
    
    def __init__(self, data, tokenizer, local_info=None, global_info=None, num_classes=2):

        self.data           = data
        self.tokenizer      = tokenizer
        self.local_info     = local_info
        self.global_info    = global_info
        self.num_classes    = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item        = self.data[idx]
        
        dial_encoding = conv_encoder(item, self.tokenizer, self.local_info, self.global_info, self.num_classes)
        
        dial_encoding['speakers'] = item['speakers']
        dial_encoding['utterances'] = item['utterances']
        dial_encoding['dialogue_id'] = item['dialogue_id']
        if self.local_info is not None:
            dial_encoding['local_info'] = item[self.local_info]
        if self.global_info is not None:
            dial_encoding['global_info'] = item[self.global_info]
        
        return dial_encoding
        
    
def get_data_loaders(
    train_data,
    dev_data,
    test_data,
    tokenizer,
    args,
    shuffle_train=True,
):
    train_set = DialogueDataset(
        train_data,  tokenizer= tokenizer, local_info = args.local_info, global_info = args.global_info, num_classes = args.num_classes
    )
    
    train_loader = DataLoader(
        train_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=True
    )

    dev_set = DialogueDataset(
        dev_data,  tokenizer= tokenizer, local_info = args.local_info, global_info = args.global_info, num_classes = args.num_classes
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    test_set = DialogueDataset(
        test_data,  tokenizer= tokenizer, local_info = args.local_info, global_info = args.global_info, num_classes = args.num_classes
    )

    test_loader = DataLoader(
        test_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    return train_loader, dev_loader, test_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CB')
    parser.add_argument('--local_info', type=str, default='intentions')
    parser.add_argument('--global_info', type=str, default='scd_summary')
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_data = json.load(open(f'../data/{args.dataset}/processed/RAT_0.5.json'))
    dev_data = json.load(open(f'../data/{args.dataset}/processed/RAT_0.5.json'))
    test_data = json.load(open(f'../data/{args.dataset}/processed/RAT_0.5.json'))

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_data,
        dev_data,
        test_data,
        tokenizer,
        args,
    )
