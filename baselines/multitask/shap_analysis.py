import shap
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import json
from transformers import AutoTokenizer
import re
from collections import defaultdict


class SHAPModelWrapper:
    """
    Wrapper class to make the HAN model compatible with SHAP explainers
    """
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.model.eval()
    
    def predict_proba(self, data_samples):
        """
        Predict probabilities for SHAP - expects list of data samples
        """
        predictions = []
        
        for sample in data_samples:
            # Convert sample to the format expected by your model
            processed_sample = self._process_sample(sample)
            
            with torch.no_grad():
                output = self.model(processed_sample)
                logits = output['logits']
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                predictions.append(probs[0])  # Remove batch dimension
        
        return np.array(predictions)
    
    def _process_sample(self, sample):
        """
        Process a single sample into the format expected by your model
        """
        # Create batch of size 1
        utt_ids = []
        utt_masks = []
        
        # Process utterances with intentions if available
        local_infos = sample.get('intentions', [None] * len(sample['speakers']))
        
        for speaker, utterance, intention in zip(sample['speakers'], sample['utterances'], local_infos):
            if intention is not None and self.args.local_info is not None:
                utt_data = f'{speaker}{self.tokenizer.sep_token}{utterance}{self.tokenizer.sep_token}{intention}'
            else:
                utt_data = f'{speaker}{self.tokenizer.sep_token}{utterance}'
                
            encoding = self.tokenizer.encode_plus(
                utt_data,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
                max_length=self.args.max_seq_len
            )
            
            utt_ids.append(encoding['input_ids'])
            utt_masks.append(encoding['attention_mask'])
        
        utt_ids = torch.cat(utt_ids, dim=0).to(self.device)
        utt_masks = torch.cat(utt_masks, dim=0).to(self.device)
        
        # Process global summary if available
        global_ids = None
        global_masks = None
        
        if self.args.global_info is not None and self.args.global_info in sample:
            global_summary = sample[self.args.global_info]
            global_encoding = self.tokenizer.encode_plus(
                global_summary,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
                max_length=self.args.max_seq_len
            )
            global_ids = global_encoding['input_ids'].to(self.device)
            global_masks = global_encoding['attention_mask'].to(self.device)
        
        return {
            "utt_ids": [utt_ids],
            "utt_masks": [utt_masks],
            "global_ids": global_ids,
            "global_masks": global_masks
        }


class PhraseImportanceAnalyzer:
    """
    Analyzer for understanding phrase-level importance in predictions
    """
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.tokenizer = model_wrapper.tokenizer
        
    def extract_phrases(self, text, phrase_length=3):
        """
        Extract overlapping phrases from text
        """
        tokens = self.tokenizer.tokenize(text)
        phrases = []
        
        # Extract n-grams of different lengths
        for n in range(1, phrase_length + 1):
            for i in range(len(tokens) - n + 1):
                phrase = self.tokenizer.convert_tokens_to_string(tokens[i:i+n])
                phrases.append({
                    'phrase': phrase.strip(),
                    'start_idx': i,
                    'end_idx': i + n,
                    'length': n,
                    'tokens': tokens[i:i+n]
                })
        
        return phrases
    
    def analyze_phrase_importance(self, sample, num_perturbations=50):
        """
        Analyze importance of phrases by masking them and measuring prediction change
        """
        # Get baseline prediction
        baseline_prob = self.model_wrapper.predict_proba([sample])[0]
        
        results = {
            'conversation_phrases': [],
            'summary_phrases': [],
            'baseline_prob': baseline_prob,
            'phrase_impacts': []
        }
        
        # Analyze conversation utterances
        for utt_idx, utterance in enumerate(sample['utterances']):
            phrases = self.extract_phrases(utterance)
            
            for phrase_info in phrases:
                # Create masked version
                masked_sample = self._mask_phrase_in_utterance(
                    sample, utt_idx, phrase_info
                )
                
                # Get prediction for masked version
                masked_prob = self.model_wrapper.predict_proba([masked_sample])[0]
                
                # Calculate impact
                impact = baseline_prob - masked_prob
                
                phrase_result = {
                    'type': 'conversation',
                    'utterance_idx': utt_idx,
                    'speaker': sample['speakers'][utt_idx],
                    'phrase': phrase_info['phrase'],
                    'phrase_length': phrase_info['length'],
                    'original_text': utterance,
                    'impact': impact,
                    'impact_magnitude': np.linalg.norm(impact),
                    'baseline_prob': baseline_prob,
                    'masked_prob': masked_prob
                }
                
                results['conversation_phrases'].append(phrase_result)
                results['phrase_impacts'].append(phrase_result)
        
        # Analyze global summary if present
        if (self.model_wrapper.args.global_info is not None and 
            self.model_wrapper.args.global_info in sample):
            
            summary_text = sample[self.model_wrapper.args.global_info]
            phrases = self.extract_phrases(summary_text)
            
            for phrase_info in phrases:
                # Create masked version
                masked_sample = self._mask_phrase_in_summary(
                    sample, phrase_info
                )
                
                # Get prediction for masked version
                masked_prob = self.model_wrapper.predict_proba([masked_sample])[0]
                
                # Calculate impact
                impact = baseline_prob - masked_prob
                
                phrase_result = {
                    'type': 'summary',
                    'phrase': phrase_info['phrase'],
                    'phrase_length': phrase_info['length'],
                    'original_text': summary_text,
                    'impact': impact,
                    'impact_magnitude': np.linalg.norm(impact),
                    'baseline_prob': baseline_prob,
                    'masked_prob': masked_prob
                }
                
                results['summary_phrases'].append(phrase_result)
                results['phrase_impacts'].append(phrase_result)
        
        return results
    
    def _mask_phrase_in_utterance(self, sample, utt_idx, phrase_info):
        """
        Create a sample with a specific phrase masked in an utterance
        """
        masked_sample = sample.copy()
        masked_sample['utterances'] = sample['utterances'].copy()
        
        # Replace the phrase with [MASK] tokens
        original_utterance = sample['utterances'][utt_idx]
        tokens = self.tokenizer.tokenize(original_utterance)
        
        # Replace tokens with [MASK]
        masked_tokens = tokens.copy()
        for i in range(phrase_info['start_idx'], phrase_info['end_idx']):
            if i < len(masked_tokens):
                masked_tokens[i] = '[MASK]'
        
        masked_utterance = self.tokenizer.convert_tokens_to_string(masked_tokens)
        masked_sample['utterances'][utt_idx] = masked_utterance
        
        return masked_sample
    
    def _mask_phrase_in_summary(self, sample, phrase_info):
        """
        Create a sample with a specific phrase masked in the summary
        """
        masked_sample = sample.copy()
        
        # Replace the phrase with [MASK] tokens in summary
        summary_key = self.model_wrapper.args.global_info
        original_summary = sample[summary_key]
        tokens = self.tokenizer.tokenize(original_summary)
        
        # Replace tokens with [MASK]
        masked_tokens = tokens.copy()
        for i in range(phrase_info['start_idx'], phrase_info['end_idx']):
            if i < len(masked_tokens):
                masked_tokens[i] = '[MASK]'
        
        masked_summary = self.tokenizer.convert_tokens_to_string(masked_tokens)
        masked_sample[summary_key] = masked_summary
        
        return masked_sample


class AttentionComparisonAnalyzer:
    """
    Analyzer for comparing attention between summaries and conversations
    """
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
    
    def compare_summary_vs_conversation_importance(self, test_samples):
        """
        Compare the overall importance of summaries vs conversations
        """
        results = []
        
        for sample_idx, sample in enumerate(test_samples):
            print(f"Analyzing sample {sample_idx + 1}/{len(test_samples)}")
            
            # Get baseline prediction (with both summary and conversation)
            baseline_prob = self.model_wrapper.predict_proba([sample])[0]
            
            # Remove summary
            sample_no_summary = sample.copy()
            if self.model_wrapper.args.global_info in sample_no_summary:
                del sample_no_summary[self.model_wrapper.args.global_info]
            
            # Temporarily disable global_info
            original_global_info = self.model_wrapper.args.global_info
            self.model_wrapper.args.global_info = None
            prob_no_summary = self.model_wrapper.predict_proba([sample_no_summary])[0]
            self.model_wrapper.args.global_info = original_global_info
            
            # Remove conversation (keep only summary)
            sample_only_summary = sample.copy()
            sample_only_summary['utterances'] = ['']  # Empty utterances
            sample_only_summary['speakers'] = ['']
            if 'intentions' in sample_only_summary:
                sample_only_summary['intentions'] = ['']
            
            prob_only_summary = self.model_wrapper.predict_proba([sample_only_summary])[0]
            
            # Calculate impacts
            summary_impact = baseline_prob - prob_no_summary
            conversation_impact = baseline_prob - prob_only_summary
            
            result = {
                'sample_idx': sample_idx,
                'baseline_prob': baseline_prob,
                'prob_no_summary': prob_no_summary,
                'prob_only_summary': prob_only_summary,
                'summary_impact': summary_impact,
                'conversation_impact': conversation_impact,
                'summary_impact_magnitude': np.linalg.norm(summary_impact),
                'conversation_impact_magnitude': np.linalg.norm(conversation_impact),
                'summary_dominance_ratio': np.linalg.norm(summary_impact) / 
                                         max(np.linalg.norm(conversation_impact), 1e-8)
            }
            
            results.append(result)
        
        return results


def visualize_phrase_importance(phrase_results, top_k=20, save_path=None):
    """
    Visualize the most important phrases
    """
    all_phrases = phrase_results['phrase_impacts']
    
    # Sort by impact magnitude
    sorted_phrases = sorted(all_phrases, key=lambda x: x['impact_magnitude'], reverse=True)
    top_phrases = sorted_phrases[:top_k]
    
    # Separate by type
    conv_phrases = [p for p in top_phrases if p['type'] == 'conversation']
    summary_phrases = [p for p in top_phrases if p['type'] == 'summary']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Top conversation phrases
    if conv_phrases:
        conv_impacts = [p['impact_magnitude'] for p in conv_phrases[:10]]
        conv_labels = [f"{p['phrase'][:30]}..." if len(p['phrase']) > 30 
                      else p['phrase'] for p in conv_phrases[:10]]
        
        axes[0, 0].barh(range(len(conv_impacts)), conv_impacts, color='lightblue')
        axes[0, 0].set_yticks(range(len(conv_impacts)))
        axes[0, 0].set_yticklabels(conv_labels)
        axes[0, 0].set_xlabel('Impact Magnitude')
        axes[0, 0].set_title('Top Conversation Phrases by Impact')
        axes[0, 0].invert_yaxis()
    
    # Plot 2: Top summary phrases
    if summary_phrases:
        summ_impacts = [p['impact_magnitude'] for p in summary_phrases[:10]]
        summ_labels = [f"{p['phrase'][:30]}..." if len(p['phrase']) > 30 
                      else p['phrase'] for p in summary_phrases[:10]]
        
        axes[0, 1].barh(range(len(summ_impacts)), summ_impacts, color='lightcoral')
        axes[0, 1].set_yticks(range(len(summ_impacts)))
        axes[0, 1].set_yticklabels(summ_labels)
        axes[0, 1].set_xlabel('Impact Magnitude')
        axes[0, 1].set_title('Top Summary Phrases by Impact')
        axes[0, 1].invert_yaxis()
    
    # Plot 3: Distribution of phrase impacts by type
    conv_magnitudes = [p['impact_magnitude'] for p in all_phrases if p['type'] == 'conversation']
    summ_magnitudes = [p['impact_magnitude'] for p in all_phrases if p['type'] == 'summary']
    
    axes[1, 0].hist(conv_magnitudes, bins=30, alpha=0.7, label='Conversation', color='lightblue')
    if summ_magnitudes:
        axes[1, 0].hist(summ_magnitudes, bins=30, alpha=0.7, label='Summary', color='lightcoral')
    axes[1, 0].set_xlabel('Impact Magnitude')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Phrase Impact Magnitudes')
    axes[1, 0].legend()
    
    # Plot 4: Phrase length vs impact
    phrase_lengths = [p['phrase_length'] for p in all_phrases]
    phrase_impacts = [p['impact_magnitude'] for p in all_phrases]
    phrase_types = [p['type'] for p in all_phrases]
    
    conv_mask = np.array(phrase_types) == 'conversation'
    summ_mask = np.array(phrase_types) == 'summary'
    
    axes[1, 1].scatter(np.array(phrase_lengths)[conv_mask], 
                      np.array(phrase_impacts)[conv_mask], 
                      alpha=0.6, label='Conversation', color='lightblue')
    if np.any(summ_mask):
        axes[1, 1].scatter(np.array(phrase_lengths)[summ_mask], 
                          np.array(phrase_impacts)[summ_mask], 
                          alpha=0.6, label='Summary', color='lightcoral')
    axes[1, 1].set_xlabel('Phrase Length (tokens)')
    axes[1, 1].set_ylabel('Impact Magnitude')
    axes[1, 1].set_title('Phrase Length vs Impact')
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_summary_vs_conversation_comparison(comparison_results, save_path=None):
    """
    Visualize the comparison between summary and conversation importance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    summary_impacts = [r['summary_impact_magnitude'] for r in comparison_results]
    conversation_impacts = [r['conversation_impact_magnitude'] for r in comparison_results]
    dominance_ratios = [r['summary_dominance_ratio'] for r in comparison_results]
    
    # Plot 1: Summary vs Conversation Impact Scatter
    axes[0, 0].scatter(conversation_impacts, summary_impacts, alpha=0.6)
    axes[0, 0].plot([0, max(max(conversation_impacts), max(summary_impacts))], 
                   [0, max(max(conversation_impacts), max(summary_impacts))], 
                   'r--', alpha=0.7)
    axes[0, 0].set_xlabel('Conversation Impact Magnitude')
    axes[0, 0].set_ylabel('Summary Impact Magnitude')
    axes[0, 0].set_title('Summary vs Conversation Impact')
    
    # Add quadrant labels
    max_val = max(max(conversation_impacts), max(summary_impacts))
    axes[0, 0].text(max_val*0.1, max_val*0.9, 'Summary\nDominant', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
    axes[0, 0].text(max_val*0.9, max_val*0.1, 'Conversation\nDominant', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    # Plot 2: Distribution of dominance ratios
    axes[0, 1].hist(dominance_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Equal Impact')
    axes[0, 1].set_xlabel('Summary Dominance Ratio (Summary Impact / Conversation Impact)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Summary Dominance')
    axes[0, 1].legend()
    
    # Plot 3: Impact magnitudes comparison
    impact_data = pd.DataFrame({
        'Impact Type': ['Summary'] * len(summary_impacts) + ['Conversation'] * len(conversation_impacts),
        'Impact Magnitude': summary_impacts + conversation_impacts
    })
    
    summary_data = impact_data[impact_data['Impact Type'] == 'Summary']['Impact Magnitude']
    conv_data = impact_data[impact_data['Impact Type'] == 'Conversation']['Impact Magnitude']
    
    axes[1, 0].boxplot([summary_data, conv_data], labels=['Summary', 'Conversation'])
    axes[1, 0].set_ylabel('Impact Magnitude')
    axes[1, 0].set_title('Impact Magnitude Distribution by Type')
    
    # Plot 4: Sample-wise comparison
    sample_indices = range(len(comparison_results))
    axes[1, 1].plot(sample_indices, summary_impacts, 'o-', label='Summary Impact', alpha=0.7)
    axes[1, 1].plot(sample_indices, conversation_impacts, 's-', label='Conversation Impact', alpha=0.7)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Impact Magnitude')
    axes[1, 1].set_title('Impact Magnitude by Sample')
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY VS CONVERSATION ANALYSIS ===")
    print(f"Average Summary Impact: {np.mean(summary_impacts):.4f}")
    print(f"Average Conversation Impact: {np.mean(conversation_impacts):.4f}")
    print(f"Average Summary Dominance Ratio: {np.mean(dominance_ratios):.4f}")
    print(f"Samples where Summary Dominates (ratio > 1): {sum(1 for r in dominance_ratios if r > 1)} / {len(dominance_ratios)}")
    print(f"Samples where Conversation Dominates (ratio < 1): {sum(1 for r in dominance_ratios if r < 1)} / {len(dominance_ratios)}")


# Enhanced main analysis function
def run_shap_analysis(model, tokenizer, device, args, test_data, background_data=None):
    """
    Run complete enhanced SHAP analysis including phrase-level importance
    """
    print("Setting up enhanced SHAP analysis...")
    
    # Create model wrapper
    model_wrapper = SHAPModelWrapper(model, tokenizer, device, args)
    
    # Use subset of test data if no background data provided
    if background_data is None:
        background_data = test_data[:min(50, len(test_data))]
    
    test_samples = test_data  # Limit for phrase analysis
    
    print("Analyzing phrase-level importance...")
    # Analyze phrase importance
    phrase_analyzer = PhraseImportanceAnalyzer(model_wrapper)
    
    phrase_results_all = []
    for i, sample in enumerate(test_samples):
        print(f"Analyzing phrases in sample {i+1}/{len(test_samples)}")
        phrase_result = phrase_analyzer.analyze_phrase_importance(sample)
        phrase_results_all.append(phrase_result)
    
    # Combine all phrase results
    combined_phrase_results = {
        'phrase_impacts': [],
        'conversation_phrases': [],
        'summary_phrases': []
    }
    
    for result in phrase_results_all:
        combined_phrase_results['phrase_impacts'].extend(result['phrase_impacts'])
        combined_phrase_results['conversation_phrases'].extend(result['conversation_phrases'])
        combined_phrase_results['summary_phrases'].extend(result['summary_phrases'])
    
    print("Analyzing summary vs conversation importance...")
    # Compare summary vs conversation importance
    attention_analyzer = AttentionComparisonAnalyzer(model_wrapper)
    comparison_results = attention_analyzer.compare_summary_vs_conversation_importance(test_samples)
    
    print("Creating visualizations...")
    # Create visualizations
    visualize_phrase_importance(combined_phrase_results, save_path='phrase_importance.png')
    visualize_summary_vs_conversation_comparison(comparison_results, save_path='summary_vs_conversation.png')
    
    return {
        'phrase_results': combined_phrase_results,
        'comparison_results': comparison_results,
        'individual_phrase_results': phrase_results_all
    }


# Example usage
if __name__ == "__main__":
    # Load your trained model
    # model = BERT_HierarchicalTransformer(args)
    # model.load_state_dict(torch.load('path_to_your_checkpoint.pt'))
    # model.to(device)
    
    # Load test data
    # test_data = json.load(open('path_to_test_data.json'))
    
    # Run enhanced analysis
    # results = run_enhanced_shap_analysis(model, tokenizer, device, args, test_data)
    
    pass