import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from transformers import AutoModel, AutoTokenizer
import argparse


class BERT_HierarchicalTransformer(nn.Module):
    def __init__(self, args):
        super(BERT_HierarchicalTransformer, self).__init__()

        self.model_name  = args.model_name
        self.global_info = args.global_info

        num_heads        = 1 
        num_layers       = 1 
        dropout          = 0.2

        self.bert                       = AutoModel.from_pretrained(self.model_name)
        self.bert_hidden_size           = self.bert.config.hidden_size
        self.transformer_hidden_size    = self.bert_hidden_size
        self.sentence_projection        = nn.Linear(self.bert_hidden_size, self.transformer_hidden_size)

        if args.trainable is False:
            for p in self.bert.parameters():
                p.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.transformer_hidden_size,
            nhead = num_heads,
            dim_feedforward= self.transformer_hidden_size * num_heads,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.global_info is not None:
            self.global_attn = nn.MultiheadAttention(embed_dim=self.bert_hidden_size, num_heads=num_heads, batch_first=True)

        # Final classification or output layer (optional)
        self.output_dim = self.transformer_hidden_size
        self.task1_classes = getattr(args, 'task1_classes', args.num_classes)  # Default to num_classes for backward compatibility
        self.task2_classes = getattr(args, 'task2_classes', args.num_classes)

        # Separate classification heads for each task
        if self.task1_classes == 1: # Regression task
            self.classifier_task1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, 1),
                nn.Sigmoid()
            )
        else:
            # Task 1: Original binary classification (donate/not donate)
            self.classifier_task1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, self.task1_classes),
            )
            
        if self.task2_classes == 1: # Regression task
            self.classifier_task2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, 1),
                nn.Sigmoid()
            )
        else:
            # Task 2: New classification (can be binary or multi-class)
            self.classifier_task2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, self.task2_classes),
            )

        # Optional: Shared representation layer before task-specific heads
        self.shared_representation = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, data):
        input_ids_batch, attention_masks_batch = data['utt_ids'], data['utt_masks']

        batch_embeddings = []

        for input_ids, attention_mask in zip(input_ids_batch, attention_masks_batch):
            # BERT encoding for each sentence
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [num_sentences, hidden]
            batch_embeddings.append(cls_embeddings)

        # Pad sequences of sentence embeddings
        padded = nn.utils.rnn.pad_sequence(batch_embeddings, batch_first=True)
        lengths = torch.tensor([x.size(0) for x in batch_embeddings], device=padded.device)

        # Sentence-level projection to Transformer dimension
        x = self.sentence_projection(padded)  # [batch_size, num_sentences, transformer_hidden_size]

        # Create padding mask for transformer (True = padding)
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

        # Pass through hierarchical transformer
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        if 'global_ids' in data and data['global_ids'] is not None and self.global_info is not None:
            global_ids_batch = data['global_ids']
            global_masks_batch = data['global_masks']
            global_summary = self.bert(input_ids=global_ids_batch, attention_mask=global_masks_batch)

            global_summary = global_summary.last_hidden_state[:,0,:].unsqueeze(1)

            global_output, _ = self.global_attn(global_summary, transformer_output, transformer_output)
            global_output = global_output.squeeze(1)
            doc_rep = global_output + transformer_output.mean(dim=1)  # [batch_size, hidden_size]
        else:
            doc_rep = transformer_output.mean(dim=1)  # [batch_size, hidden_size]

        # Optional: Pass through shared representation layer
        shared_rep = self.shared_representation(doc_rep)
        
        # Task-specific classification heads
        output_task1 = self.classifier_task1(shared_rep)
        output_task2 = self.classifier_task2(shared_rep)
        
        return {
            'logits_task1': output_task1,
            'logits_task2': output_task2,
            'doc_rep': doc_rep,
            'shared_rep': shared_rep
        }


# Alternative version with task-specific representation layers
class BERT_HierarchicalTransformer_TaskSpecific(nn.Module):
    """
    Alternative implementation where each task has its own representation layer
    after the shared transformer encoder. This can help when tasks are quite different.
    """
    def __init__(self, args):
        super(BERT_HierarchicalTransformer_TaskSpecific, self).__init__()

        self.model_name  = args.model_name
        self.global_info = args.global_info

        num_heads        = 1 
        num_layers       = 1 
        dropout          = 0.2

        self.bert                       = AutoModel.from_pretrained(self.model_name)
        self.bert_hidden_size           = self.bert.config.hidden_size
        self.transformer_hidden_size    = self.bert_hidden_size
        self.sentence_projection        = nn.Linear(self.bert_hidden_size, self.transformer_hidden_size)

        if args.trainable is False:
            for p in self.bert.parameters():
                p.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.transformer_hidden_size,
            nhead = num_heads,
            dim_feedforward= self.transformer_hidden_size * num_heads,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.global_info is not None:
            self.global_attn = nn.MultiheadAttention(embed_dim=self.bert_hidden_size, num_heads=num_heads, batch_first=True)

        self.output_dim = self.transformer_hidden_size
        self.task1_classes = getattr(args, 'task1_classes', args.num_classes)
        self.task2_classes = getattr(args, 'task2_classes', args.num_classes)

        # Task-specific representation layers
        self.task1_representation = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.task2_representation = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Task-specific classification heads
        if self.task1_classes == 1:
            self.classifier_task1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier_task1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, self.task1_classes),
            )
            
        if self.task2_classes == 1:
            self.classifier_task2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier_task2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, self.task2_classes),
            )
        
    def forward(self, data):
        input_ids_batch, attention_masks_batch = data['utt_ids'], data['utt_masks']

        batch_embeddings = []

        for input_ids, attention_mask in zip(input_ids_batch, attention_masks_batch):
            # BERT encoding for each sentence
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [num_sentences, hidden]
            batch_embeddings.append(cls_embeddings)

        # Pad sequences of sentence embeddings
        padded = nn.utils.rnn.pad_sequence(batch_embeddings, batch_first=True)
        lengths = torch.tensor([x.size(0) for x in batch_embeddings], device=padded.device)

        # Sentence-level projection to Transformer dimension
        x = self.sentence_projection(padded)  # [batch_size, num_sentences, transformer_hidden_size]

        # Create padding mask for transformer (True = padding)
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

        # Pass through hierarchical transformer
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        if 'global_ids' in data and data['global_ids'] is not None and self.global_info is not None:
            global_ids_batch = data['global_ids']
            global_masks_batch = data['global_masks']
            global_summary = self.bert(input_ids=global_ids_batch, attention_mask=global_masks_batch)

            global_summary = global_summary.last_hidden_state[:,0,:].unsqueeze(1)

            global_output, _ = self.global_attn(global_summary, transformer_output, transformer_output)
            global_output = global_output.squeeze(1)
            doc_rep = global_output + transformer_output.mean(dim=1)  # [batch_size, hidden_size]
        else:
            doc_rep = transformer_output.mean(dim=1)  # [batch_size, hidden_size]

        # Task-specific representations
        task1_rep = self.task1_representation(doc_rep)
        task2_rep = self.task2_representation(doc_rep)
        
        # Task-specific classifications
        output_task1 = self.classifier_task1(task1_rep)
        output_task2 = self.classifier_task2(task2_rep)
        
        return {
            'logits_task1': output_task1,
            'logits_task2': output_task2,
            'doc_rep': doc_rep,
            'task1_rep': task1_rep,
            'task2_rep': task2_rep
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--global_info', type=str, default='traditional_summary')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--trainable', type=bool, default=True)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Example: A batch of 2 documents with sentence lists
    docs = [
        ["Hello world.", "This is the second sentence."],
        ["Another document with", "three sentences in it.", "Nice!"]
    ]

    summary = [
        'This is a summary of the first document.',
        'This is a summary of the second document.'
    ]

    input_ids_batch = []
    attention_masks_batch = []

    summary_ids_batch = []
    summary_masks_batch = []

    for sentences in docs:
        encoded = [tokenizer(s, return_tensors="pt", padding="max_length", truncation=True, max_length=32) for s in sentences]
        input_ids = torch.cat([e["input_ids"] for e in encoded], dim=0)
        attention_masks = torch.cat([e["attention_mask"] for e in encoded], dim=0)
        
        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attention_masks)
    
    for summ in summary:
        encoded = tokenizer(summ, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

        summary_ids_batch.append(encoded["input_ids"])
        summary_masks_batch.append(encoded["attention_mask"])

    data = {
        'utt_ids': input_ids_batch,
        'utt_masks': attention_masks_batch,
        'global_ids': torch.vstack(summary_ids_batch),
        'global_masks': torch.vstack(summary_masks_batch)
    }

    model = BERT_HierarchicalTransformer(args)
    output = model(data)

    print("Task 1 logits shape:", output['logits_task1'].shape)  # [batch_size, num_classes]
    print("Task 2 logits shape:", output['logits_task2'].shape)  # [batch_size, num_classes]
    print("Document representation shape:", output['doc_rep'].shape)  # [batch_size, hidden_size]