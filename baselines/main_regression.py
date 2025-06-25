import numpy as np
import math
import argparse
import json
import pandas as pd
from pprint import pprint
import os
from tqdm import tqdm
import wandb
from shap_analysis import run_shap_analysis
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
    set_seed
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
    parser.add_argument('--dataset',        type=str, default='craigslistbargain', help='Choosing the CONV Forecasting dataset')
    parser.add_argument('--input_dir',      type=str, default='baselines/data/', help='The input directory')
    parser.add_argument('--local_info',     type=str, default=None, help='Choice of local_information to use')
    parser.add_argument('--global_info',    type=str, default=None, help='Choice of global_information to use')
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
    parser.add_argument('--seed',           type=int, default=42, help='The random seed')
    parser.add_argument('--learning_rate',  type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='The number of gradient accumulation steps')
    parser.add_argument('--patience',       type=int, default=5, help='The number of patience steps')
    
    parser.add_argument('--run_shap',           type=int, default=0, help='Whether to run SHAP analysis')
    parser.add_argument('--shap_samples',       type=int, default=100, help='Number of samples for SHAP analysis')
    parser.add_argument('--shap_background',    type=int, default=50, help='Number of background samples for SHAP')
    
    args = parser.parse_args()
    return args


def seen_eval(model, data_loader, device, args, tokenizer, save_csv=False, csv_path=None):
    model.eval()
    targets_all = []
    total_loss = 0.0
    num_batches = 0
    per_summary_rows = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            try:
                upd_data = {
                    "utt_ids": [elem.to(device) for elem in data["utt_ids"]],
                    "utt_masks": [elem.to(device) for elem in data["utt_masks"]]
                }
                if data["global_ids"] is not None:
                    upd_data["global_ids"] = data["global_ids"].to(device)
                    upd_data["global_masks"] = data["global_masks"].to(device)

                targets = data["score"].to(device)
                output_dict = model(upd_data)
                logits = output_dict['logits']

                # Reshape both to have the same dimensions
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                # Ensure they have the same shape
                logits = logits.view(-1)
                targets = targets.view(-1)

                loss = MSE_loss(logits, targets)

                total_loss += loss.item()
  
                targets_all += targets.tolist()
                num_batches += 1

                if save_csv:
                    for j in range(len(targets)):
                        row = {
                            "dialogue_id": data.get("dialogue_id", [None])[j],
                            "utterance": ' '.join(data.get("utt_text", [None])[j]),
                            "gold_score": targets[j].item(),
                            "predicted_score": logits[j].item()
                        }
                        per_summary_rows.append(row)

            except Exception as e:  
                print(f"ERROR in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                
                if "out of memory" in str(e):
                    print(f"[OOM] Skipping eval batch {i}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Non-OOM error in batch {i}, continuing...")
                    continue

    avg_loss = total_loss / max(1, num_batches)

    # Save predictions to CSV
    if save_csv and csv_path:
        try:
            print(f"Attempting to save {len(per_summary_rows)} rows to CSV...")
            if per_summary_rows:  # Check if we have data
                df = pd.DataFrame(per_summary_rows)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # Create directory
                df.to_csv(csv_path, index=False)
                print(f"Successfully saved per-summary results to: {csv_path}")
            else:
                print("No data to save - per_summary_rows is empty")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    return {"loss": avg_loss}


if __name__ == '__main__':

    args = get_arguments()
    pprint(args)

    if args.dataset == 'cd' or args.dataset == 'p4g':
        args.num_classes = 2
    else:
        args.num_classes = 1

    seed_everything(args.seed)

    # Load the data 
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = json.load(open(f'baselines/data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_train.json'))
    dev_data = json.load(open(f'baselines/data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_test.json'))
    test_data = json.load(open(f'baselines/data/{args.dataset}/processed/RAT_{args.frac}_{args.fold}_test.json'))

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_data, dev_data, test_data, tokenizer, args,
    )

    model = BERT_HierarchicalTransformer(args)
    model.to(device)
    MSE_loss = nn.MSELoss()

    identifiable_file = f'{args.dataset}_regression_ratio_{args.frac}_{args.local_info}_{args.global_info}_{args.model_name}_fold_{args.fold}.txt'
    checkpoint_file = f'/data/user_data/rithviks/ckpts/{identifiable_file}.pt'


    if args.do_train == 1:

        #######################################
        # TRAINING LOOP                       #
        #######################################
        print("Setting up training loop...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best_mse = torch.inf
        kill_cnt = 0


        for epoch in range(args.epochs):
            print(f"============== TRAINING ON EPOCH {epoch} ==============")
            running_loss = 0.0
            model.train()

            for i, data in enumerate(tqdm(train_loader)):
                # load the data from the data loader

                upd_data           = {
                    "utt_ids":      [elem.to(device) for elem in data["utt_ids"]],
                    "utt_masks":     [elem.to(device) for elem in data["utt_masks"]]
                }
                if data["global_ids"] is not None:
                    upd_data["global_ids"] = data["global_ids"].to(device)
                    upd_data["global_masks"] = data["global_masks"].to(device)

                targets             = data["score"].to(device)

                # forward pass
                optimizer.zero_grad()
                output_dict         = model(upd_data)

                # compute the mean squared error loss between y_trues and logits
                # import pdb; pdb.set_trace()

                logits              = output_dict['logits']
                loss                = MSE_loss(logits.squeeze(), targets)

                loss.backward()
                running_loss += loss.item()
                if (i + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()

            # final gradient step if we hadn't handled it already
            if (i + 1) % args.grad_accumulation_steps != 0:
                optimizer.step()

            print("============== EVALUATION ==============")

            results = seen_eval(model, train_loader, device, args, tokenizer)
            
            print(f"Train data \t Loss: {results['loss']}")

            results = seen_eval(model, dev_loader, device, args, tokenizer)

            mse = results['loss']
            print(f"Eval data \t Loss: {results['loss']}")

            if mse < best_mse:
                kill_cnt = 0
                best_mse = mse
                torch.save(model.state_dict(), checkpoint_file)
            else:
                kill_cnt += 1
                if kill_cnt >= args.patience:
                    break
            
            print(f"[best val] mse: mse score: {best_mse:.4f}")

        print("============== EVALUATION ON TEST DATA ==============")
        model.load_state_dict(torch.load(checkpoint_file))
        model.to(device)
        model.eval()

        csv_path = f"baselines/results/{args.dataset}/seed_{args.seed}/{identifiable_file}_test.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        results = seen_eval(
            model, 
            test_loader, 
            device, 
            args, 
            tokenizer, 
            save_csv=True, 
            csv_path=csv_path
        )

        print(f"Test data \t Loss: {results['loss']}")

    # SHAP Analysis - Add this new section
    if args.run_shap == 1:
        print("============== STARTING SHAP ANALYSIS ==============")
        
        # Load the best model
        if os.path.exists(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file))
            model.to(device)
            model.eval()
            print(f"Loaded model from {checkpoint_file}")
        else:
            print("No checkpoint found, using current model state")
        
        # Prepare data for SHAP analysis
        test_samples = test_data[:args.shap_samples]
        background_samples = train_data[:args.shap_background]
        
        # Run SHAP analysis
        try:
            shap_results = run_shap_analysis(
                model=model,
                tokenizer=tokenizer,
                device=device,
                args=args,
                test_data=test_samples,
                background_data=background_samples
            )
            
            # Save SHAP results
            shap_results_dir = f"baselines/shap_results/{args.dataset}/seed_{args.seed}"
            os.makedirs(shap_results_dir, exist_ok=True)
            
            # Save intention impact results
            intention_impact = shap_results['intention_results']['intention_impact']
            np.save(f"{shap_results_dir}/intention_impact_{identifiable_file}.npy", intention_impact)
            
            # Save summary of results
            with open(f"{shap_results_dir}/shap_summary_{identifiable_file}.txt", 'w') as f:
                f.write(f"SHAP Analysis Results for {args.dataset}\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Local Info: {args.local_info}\n")
                f.write(f"Global Info: {args.global_info}\n")
                f.write(f"Samples Analyzed: {len(test_samples)}\n")
                f.write(f"Background Samples: {len(background_samples)}\n\n")
                
                # Intention impact summary
                mean_intention_impact = np.mean(intention_impact[:, 1])  # Positive class impact
                f.write(f"Mean Intention Impact on Positive Class: {mean_intention_impact:.4f}\n")
                f.write(f"Std Intention Impact: {np.std(intention_impact[:, 1]):.4f}\n")
                
                # Summary impact results
                if 'summary_results' in shap_results:
                    f.write(f"\nGlobal Summary Impact Results:\n")
                    summary_types = ['traditional_summary', 'relational_summary', 'scm_summary', 
                                   'scd_summary', 'politeness_summary']
                    
                    for summary_type in summary_types:
                        if summary_type in shap_results['summary_results'] and \
                           'impact' in shap_results['summary_results'][summary_type]:
                            impact = shap_results['summary_results'][summary_type]['impact'][:, 1]
                            mean_impact = np.mean(impact)
                            f.write(f"{summary_type}: {mean_impact:.4f}\n")
            
            print(f"SHAP analysis completed. Results saved to {shap_results_dir}")
            
            # Print key findings
            print("\n============== SHAP ANALYSIS SUMMARY ==============")
            print(f"Mean Intention Impact: {mean_intention_impact:.4f}")
            
            if 'summary_results' in shap_results:
                print("\nGlobal Summary Impacts:")
                for summary_type in summary_types:
                    if summary_type in shap_results['summary_results'] and \
                       'impact' in shap_results['summary_results'][summary_type]:
                        impact = shap_results['summary_results'][summary_type]['impact'][:, 1]
                        mean_impact = np.mean(impact)
                        print(f"  {summary_type}: {mean_impact:.4f}")
            
        except Exception as e:
            print(f"Error during SHAP analysis: {str(e)}")
            import traceback
            traceback.print_exc()