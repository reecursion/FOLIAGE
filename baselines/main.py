import numpy as np
import math
import argparse
import json
import pandas as pd
from pprint import pprint
import os
from tqdm import tqdm
import wandb

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from dataloader import *
from models import *
import transformers


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='CB', help='Choosing the CONV Forecasting dataset')
    parser.add_argument('--input_dir',      type=str, default='../data/', help='The input directory')
    parser.add_argument('--local_info',     type=str, default='intentions', help='Choice of local_information to use')
    parser.add_argument('--global_info',    type=str, default='scd_summary', help='Choice of global_information to use')
    parser.add_argument('--LLM',            type=str, default='gpt-4o', help='The language model to use')
    parser.add_argument('--frac',           type=str, default='ALL', help='The fraction of data to use')
    parser.add_argument('--fold',           type=int, default=0, help='The fold to use')

    parser.add_argument('--model_name',     type=str, default='bert-base-uncased', help='The model to use') # backbone model to use

    parser.add_argument('--trainable',      type=bool, default=True, help='Whether to train the model')
    parser.add_argument('--do_train',       type=int, default=1, help='Whether to train the model')
    parser.add_argument('--do_test',        type=int, default=1, help='Whether to test the model')
    parser.add_argument('--max_seq_len',    type=int, default=512, help='The maximum sequence length')
    parser.add_argument('--batch_size',     type=int, default=4, help='The batch size')
    parser.add_argument('--gpu',            type=str, default='0', help='The gpu to use')

    parser.add_argument('--epochs',         type=int, default=15, help='The number of training epochs')
    parser.add_argument('--seed',           type=int, default=15232, help='The random seed')
    parser.add_argument('--learning_rate',  type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='The number of gradient accumulation steps')
    parser.add_argument('--patience',       type=int, default=5, help='The number of patience steps')
    
    args = parser.parse_args()
    return args


def seen_eval(model, data_loader, device, args, tokenizer):

    model.eval()

    y_preds, y_trues = [], []
    tot_loss = 0

    for i, data in enumerate(tqdm(data_loader)):
        # load the data from the data loader

        upd_data           = {
            "utt_ids":      [elem.to(device) for elem in data["utt_ids"]],
            "utt_masks":     [elem.to(device) for elem in data["utt_masks"]],
            "global_ids":    data["global_ids"].to(device),
            "global_masks":  data["global_masks"].to(device),
        }

        targets             = data["binary_score"].to(device)
        # forward pass
        optimizer.zero_grad()
        output_dict         = model(upd_data)

        logits              = output_dict['logits']
        loss                = CE_loss(logits, targets)

        y_pred             = logits.argmax(dim=1).cpu().numpy()
        y_true             = targets.cpu().numpy()

        tot_loss += loss.item()

        y_preds.append(y_pred)
        y_trues.append(y_true)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'precision': p,
        'recall': r,
        'f1': f1,
        'accuracy': acc,
        'y_preds': y_preds,
        'y_trues': y_trues,
        'loss': tot_loss / len(data_loader)
    }

if __name__ == '__main__':

    args = get_arguments()
    pprint(args)

    if args.dataset == 'CB':
        args.num_classes = 2


    seed_everything(args.seed)

    # Load the data 

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device          = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer       = AutoTokenizer.from_pretrained(args.model_name)

    train_data      = json.load(open(f'../data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_train.json'))
    dev_data        = json.load(open(f'../data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_test.json'))
    test_data       = json.load(open(f'../data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_test.json'))

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_data,
        dev_data,
        test_data,
        tokenizer,
        args,
    )

    model   = BERT_HierarchicalTransformer(args)
    model.to(device)

    CE_loss = nn.CrossEntropyLoss()

    identifiable_file = f'{args.dataset}_{args.LLM}_{args.local_info}_{args.global_info}_{args.frac}_{args.model_name}_{args.trainable}_{args.seed}_{args.fold}'


    if args.do_train == 1:

        #######################################
        # TRAINING LOOP                       #
        #######################################
        print("Setting up training loop...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best_p, best_r, best_f1 = 0, 0, 0
        kill_cnt = 0

        checkpoint_file = f'../ckpts/{identifiable_file}.pt'

        for epoch in range(args.epochs):
            print(f"============== TRAINING ON EPOCH {epoch} ==============")
            running_loss = 0.0
            model.train()

            for i, data in enumerate(tqdm(train_loader)):
                # load the data from the data loader

                upd_data           = {
                    "utt_ids":      [elem.to(device) for elem in data["utt_ids"]],
                    "utt_masks":     [elem.to(device) for elem in data["utt_masks"]],
                    "global_ids":    data["global_ids"].to(device),
                    "global_masks":  data["global_masks"].to(device),
                }

                targets             = data["binary_score"].to(device)
                # forward pass
                optimizer.zero_grad()
                output_dict         = model(upd_data)

                # compute the binary cross entropy loss between y_trues and logits
                # import pdb; pdb.set_trace()

                logits              = output_dict['logits']

                loss                = CE_loss(logits, targets)

                loss.backward()
                running_loss += loss.item()
                if (i + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()

            # final gradient step if we hadn't handled it already
            if (i + 1) % args.grad_accumulation_steps != 0:
                optimizer.step()

            print("============== EVALUATION ==============")

            results = seen_eval(model, train_loader, device, args, tokenizer)

            p_tr, r_tr, f1_tr = results['precision'], results['recall'], results['f1']
            
            print(f"Train data \t Precision: {p_tr} \t Recall: {r_tr} \t F1: {f1_tr} \t Loss: {results['loss']}")


            results = seen_eval(model, dev_loader, device, args, tokenizer)

            p_dev, r_dev, f1_dev = results['precision'], results['recall'], results['f1']
            
            print(f"Eval data \t Precision: {p_dev} \t Recall: {r_dev} \t F1: {f1_dev}\t Loss: {results['loss']}")

            if f1_dev > best_f1:
                kill_cnt = 0
                best_p, best_r, best_f1 = p_dev, r_dev, f1_dev
                torch.save(model.state_dict(), checkpoint_file)
            else:
                kill_cnt += 1
                if kill_cnt >= args.patience:
                    break
            
            print(f"[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}")

        print("============== EVALUATION ON TEST DATA ==============")
        model.load_state_dict(torch.load(checkpoint_file))
        model.to(device)
        model.eval()

        results = seen_eval(model, test_loader, device, args, tokenizer)

        p_test, r_test, f1_test = results['precision'], results['recall'], results['f1']
        
        print(f"Test data \t Precision: {p_test} \t Recall: {r_test} \t F1: {f1_test}\t Loss: {results['loss']}")



### stop thinking about improving performance, but more about improving efficiency. 
### create a broader picture of the issue; where it works, why it doesn't work and why does it not work?

