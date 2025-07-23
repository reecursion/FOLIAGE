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
    new_label       = [s["new_label"] for s in samples]  # Add new label
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
        "new_label": torch.LongTensor(new_label),  # Add new label tensor
        "utt_text": utterances,
        "speakers": speakers,
        "dialogue_id": dialogue_id,
    }


def conv_encoder(item, tokenizer, local_info=None, global_info=None, task1_classes=2, task2_classes=2):
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
        
        dial_encoding = tokenizer.encode_plus(
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
        global_encoding = tokenizer.encode_plus(
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
    
    binary_score = item['binary_score']
    new_label = item['new_label']  # Extract new label from data

    result = {
        'utt_ids': utt_ids,
        'utt_masks': utt_mask,
        'global_ids': global_ids,
        'global_masks': global_mask,
        'binary_score': binary_score,
        'new_label': new_label,  # Add new label to result
    }

    return result


class DialogueDataset(Dataset):
    
    def __init__(self, data, tokenizer, local_info=None, global_info=None, task1_classes=2, task2_classes=2):
        self.data           = data
        self.tokenizer      = tokenizer
        self.local_info     = local_info
        self.global_info    = global_info
        self.task1_classes  = task1_classes
        self.task2_classes  = task2_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        dial_encoding = conv_encoder(item, self.tokenizer, self.local_info, self.global_info, self.task1_classes, self.task2_classes)
        
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
        train_data,  tokenizer= tokenizer, local_info = args.local_info, global_info = args.global_info, 
        task1_classes = getattr(args, 'task1_classes', args.num_classes), 
        task2_classes = getattr(args, 'task2_classes', args.num_classes)
    )
    
    train_loader = DataLoader(
        train_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=True
    )

    dev_set = DialogueDataset(
        dev_data,  tokenizer= tokenizer, local_info = args.local_info, global_info = args.global_info, 
        task1_classes = getattr(args, 'task1_classes', args.num_classes), 
        task2_classes = getattr(args, 'task2_classes', args.num_classes)
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    test_set = DialogueDataset(
        test_data,  tokenizer= tokenizer, local_info = args.local_info, global_info = args.global_info, 
        task1_classes = getattr(args, 'task1_classes', args.num_classes), 
        task2_classes = getattr(args, 'task2_classes', args.num_classes)
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