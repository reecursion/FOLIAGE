import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelPersuasionComparison:
    def __init__(self, json_file_path: str, csv_file_path: str):
        """
        Initialize with persuasion analysis JSON and model results CSV
        """
        # Load persuasion analysis
        with open(json_file_path, 'r') as f:
            self.persuasion_data = json.load(f)
        
        # Load model results
        self.model_results = pd.read_csv(csv_file_path)
        
        # Process persuasion analysis
        self.persuasion_df = self._process_persuasion_data()
        
        # Merge datasets
        self.merged_data = self._merge_datasets()
        
    def _process_persuasion_data(self) -> pd.DataFrame:
        """
        Convert persuasion analysis results to DataFrame
        """
        detailed_results = self.persuasion_data.get('detailed_results', [])
        
        # Extract relevant information
        processed_data = []
        for result in detailed_results:
            if 'error' not in result:
                processed_data.append({
                    'dialogue_id': result.get('dialogue_id'),
                    'stated_intention': result.get('stated_intention'),
                    'actual_donation': result.get('actual_donation'),
                    'inconsistency': result.get('inconsistency', False),
                    'decision_percentage': result.get('decision_percentage'),
                    'decision_utterance_idx': result.get('decision_utterance_idx'),
                    'conversation_length': result.get('conversation_length'),
                    'final_donation_made': result.get('final_donation_made'),
                    'decision_quote': result.get('decision_quote'),
                    'analysis_notes': result.get('analysis_notes')
                })
        
        return pd.DataFrame(processed_data)
    
    def _merge_datasets(self) -> pd.DataFrame:
        """
        Merge persuasion analysis with model results
        """
        # Merge on dialogue_id
        merged = pd.merge(
            self.model_results, 
            self.persuasion_df, 
            on='dialogue_id', 
            how='inner'
        )
        
        return merged
    
    def categorize_users(self) -> Dict:
        """
        Categorize users based on their promise-keeping behavior
        """
        categories = {
            'promise_keepers': [],      # Said donate + actually donated
            'promise_breakers': [],     # Said donate + didn't donate
            'refusal_reversers': [],    # Said no + actually donated
            'consistent_refusers': [],  # Said no + didn't donate
            'unclear_decision': []      # Unclear stated intention
        }
        
        for _, row in self.merged_data.iterrows():
            stated = row.get('stated_intention')
            actual = row.get('actual_donation', False)
            
            if stated == 'donate':
                if actual:
                    categories['promise_keepers'].append(row)
                else:
                    categories['promise_breakers'].append(row)
            elif stated == 'not_donate':
                if actual:
                    categories['refusal_reversers'].append(row)
                else:
                    categories['consistent_refusers'].append(row)
            else:
                categories['unclear_decision'].append(row)
        
        # Convert to DataFrames for easier analysis
        for key in categories:
            categories[key] = pd.DataFrame(categories[key])
        
        return categories
    
    def calculate_performance_by_category(self, categories: Dict) -> Dict:
        """
        Calculate model performance metrics for each user category
        """
        performance = {}
        
        for category_name, category_df in categories.items():
            if len(category_df) == 0:
                performance[category_name] = {
                    'count': 0,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'avg_confidence': None
                }
                continue
            
            # Calculate metrics
            y_true = category_df['label']
            y_pred = category_df['predicted_label']
            correct = category_df['correct']
            confidence = category_df['confidence_score']
            
            performance[category_name] = {
                'count': len(category_df),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'avg_confidence': confidence.mean(),
                'correct_predictions': correct.sum(),
                'label_distribution': y_true.value_counts().to_dict(),
                'prediction_distribution': y_pred.value_counts().to_dict()
            }
        
        return performance
    
    def calculate_aggregate_metrics(self) -> Dict:
        """
        Calculate overall aggregate metrics across all conversations
        """
        if len(self.merged_data) == 0:
            return {"error": "No merged data available"}
        
        y_true = self.merged_data['label']
        y_pred = self.merged_data['predicted_label']
        
        aggregate_metrics = {
            'total_conversations': len(self.merged_data),
            'overall_accuracy': accuracy_score(y_true, y_pred),
            'overall_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'overall_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'overall_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'overall_macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'overall_micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'average_confidence': self.merged_data['confidence_score'].mean(),
            'correct_predictions': self.merged_data['correct'].sum(),
            'unique_labels': sorted(y_true.unique().tolist()),
            'label_distribution': y_true.value_counts().to_dict(),
            'prediction_distribution': y_pred.value_counts().to_dict()
        }
        
        return aggregate_metrics
    
    def create_promise_confusion_matrix(self, categories: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Create confusion matrix specifically for promise-keeping behavior
        """
        # Combine promise keepers and breakers
        promise_data = []
        
        # Promise keepers (said yes, donated) -> True promise kept
        if len(categories['promise_keepers']) > 0:
            pk_data = categories['promise_keepers'].copy()
            pk_data['promise_kept_actual'] = True
            promise_data.append(pk_data)
        
        # Promise breakers (said yes, didn't donate) -> False promise kept
        if len(categories['promise_breakers']) > 0:
            pb_data = categories['promise_breakers'].copy()
            pb_data['promise_kept_actual'] = False
            promise_data.append(pb_data)
        
        if not promise_data:
            return None, {"error": "No promise data available"}
        
        promise_df = pd.concat(promise_data, ignore_index=True)
        
        # Map model predictions to promise-keeping predictions
        def map_prediction_to_promise_kept(row):
            # If prediction indicates donation, we predict promise will be kept
            pred_label = row['predicted_label']
            confidence = row['confidence_score']
            
            # Simple mapping - adjust based on your label scheme
            if pred_label == 1 or pred_label == 'donate' or pred_label == True:
                return True
            elif pred_label == 0 or pred_label == 'no_donate' or pred_label == False:
                return False
            else:
                # For multi-class, use confidence threshold or most likely donation outcome
                return confidence > 0.5
        
        promise_df['promise_kept_predicted'] = promise_df.apply(map_prediction_to_promise_kept, axis=1)
        
        # Create confusion matrix
        y_true_promise = promise_df['promise_kept_actual']
        y_pred_promise = promise_df['promise_kept_predicted']
        
        cm = confusion_matrix(y_true_promise, y_pred_promise)
        
        # Calculate metrics
        promise_metrics = {
            'total_promise_conversations': len(promise_df),
            'promise_accuracy': accuracy_score(y_true_promise, y_pred_promise),
            'promise_precision': precision_score(y_true_promise, y_pred_promise, zero_division=0),
            'promise_recall': recall_score(y_true_promise, y_pred_promise, zero_division=0),
            'promise_f1': f1_score(y_true_promise, y_pred_promise, zero_division=0),
            'promise_keepers_count': (y_true_promise == True).sum(),
            'promise_breakers_count': (y_true_promise == False).sum(),
            'predicted_promise_kept_count': (y_pred_promise == True).sum(),
            'predicted_promise_broken_count': (y_pred_promise == False).sum()
        }
        
        return cm, promise_metrics
    
    def plot_promise_confusion_matrix(self, promise_cm: np.ndarray, promise_metrics: Dict, 
                                     save_path: str = 'promise_confusion_matrix.png'):
        """
        Create a detailed confusion matrix plot specifically for promise-keeping behavior
        """
        if promise_cm is None or promise_metrics is None:
            print("No promise confusion matrix data available")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Confusion Matrix Heatmap
        im = ax1.imshow(promise_cm, interpolation='nearest', cmap='Blues')
        ax1.figure.colorbar(im, ax=ax1)
        
        # Set labels and title
        classes = ['Promise Broken', 'Promise Kept']
        ax1.set(xticks=np.arange(promise_cm.shape[1]),
                yticks=np.arange(promise_cm.shape[0]),
                xticklabels=classes,
                yticklabels=classes,
                title=f'Promise Confusion Matrix\n(Accuracy: {promise_metrics["promise_accuracy"]:.3f})',
                ylabel='Actual',
                xlabel='Predicted')
        
        # Add text annotations
        thresh = promise_cm.max() / 2.
        for i in range(promise_cm.shape[0]):
            for j in range(promise_cm.shape[1]):
                ax1.text(j, i, format(promise_cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if promise_cm[i, j] > thresh else "black",
                        fontsize=16, fontweight='bold')
        
        # Add percentage annotations
        total = promise_cm.sum()
        for i in range(promise_cm.shape[0]):
            for j in range(promise_cm.shape[1]):
                percentage = promise_cm[i, j] / total * 100
                ax1.text(j, i-0.25, f'({percentage:.1f}%)',
                        ha="center", va="center",
                        color="white" if promise_cm[i, j] > thresh else "black",
                        fontsize=10)
        
        # Plot 2: Metrics Bar Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [
            promise_metrics['promise_accuracy'],
            promise_metrics['promise_precision'],
            promise_metrics['promise_recall'],
            promise_metrics['promise_f1']
        ]
        
        bars = ax2.bar(metrics_names, metrics_values, 
                      color=['skyblue', 'lightgreen', 'orange', 'lightcoral'],
                      alpha=0.7)
        
        ax2.set_ylim(0, 1)
        ax2.set_title('Promise Prediction Metrics')
        ax2.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add sample size info
        total_promises = promise_metrics['total_promise_conversations']
        keepers = promise_metrics['promise_keepers_count']
        breakers = promise_metrics['promise_breakers_count']
        
        ax2.text(0.02, 0.95, f'Total Promise Conversations: {total_promises}\n'
                            f'Promise Keepers: {keepers}\n'
                            f'Promise Breakers: {breakers}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed analysis
        print(f"\n" + "="*60)
        print("PROMISE CONFUSION MATRIX ANALYSIS")
        print("="*60)
        
        tn, fp, fn, tp = promise_cm.ravel()
        
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Negatives (Correctly predicted broken): {tn}")
        print(f"False Positives (Predicted kept, actually broken): {fp}")
        print(f"False Negatives (Predicted broken, actually kept): {fn}")
        print(f"True Positives (Correctly predicted kept): {tp}")
        
        print(f"\nError Analysis:")
        if fp > fn:
            print(f"• Model is overly OPTIMISTIC ({fp} vs {fn})")
            print("• Tends to predict people will keep promises when they won't")
            print("• May miss signals of potential promise-breaking")
        elif fn > fp:
            print(f"• Model is overly PESSIMISTIC ({fn} vs {fp})")
            print("• Tends to predict people will break promises when they won't")
            print("• May be too cautious about promise-keeping predictions")
        else:
            print("• Model has balanced error types")
        
        # Calculate specific error rates
        total_actual_broken = tn + fp
        total_actual_kept = fn + tp
        
        if total_actual_broken > 0:
            false_positive_rate = fp / total_actual_broken
            print(f"\nFalse Positive Rate: {false_positive_rate:.3f}")
            print(f"  → {false_positive_rate:.1%} of promise breakers were incorrectly predicted as keepers")
        
        if total_actual_kept > 0:
            false_negative_rate = fn / total_actual_kept
            print(f"False Negative Rate: {false_negative_rate:.3f}")
            print(f"  → {false_negative_rate:.1%} of promise keepers were incorrectly predicted as breakers")
        
        return fig
    
    def analyze_decision_timing_impact(self, categories: Dict) -> Dict:
        """
        Analyze how decision timing affects model performance
        """
        timing_analysis = {}
        
        for category_name, category_df in categories.items():
            if len(category_df) == 0 or 'decision_percentage' not in category_df.columns:
                continue
                
            # Filter out rows with missing decision percentages
            valid_decisions = category_df.dropna(subset=['decision_percentage'])
            
            if len(valid_decisions) == 0:
                continue
            
            # Bin decision timing (early, middle, late)
            valid_decisions = valid_decisions.copy()
            valid_decisions['decision_timing_bin'] = pd.cut(
                valid_decisions['decision_percentage'], 
                bins=[0, 33, 66, 100], 
                labels=['Early (0-33%)', 'Middle (33-66%)', 'Late (66-100%)']
            )
            
            timing_performance = {}
            for timing_bin in valid_decisions['decision_timing_bin'].unique():
                if pd.isna(timing_bin):
                    continue
                    
                bin_data = valid_decisions[valid_decisions['decision_timing_bin'] == timing_bin]
                
                timing_performance[str(timing_bin)] = {
                    'count': len(bin_data),
                    'accuracy': accuracy_score(bin_data['label'], bin_data['predicted_label']),
                    'avg_confidence': bin_data['confidence_score'].mean(),
                    'avg_decision_percentage': bin_data['decision_percentage'].mean()
                }
            
            timing_analysis[category_name] = timing_performance
        
        return timing_analysis
    
    def create_performance_comparison_plot(self, performance: Dict, aggregate_metrics: Dict, 
                                         promise_cm: np.ndarray = None, promise_metrics: Dict = None):
        """
        Create visualization comparing model performance across categories
        """
        # Prepare data for plotting
        categories = []
        accuracies = []
        f1_scores = []
        counts = []
        confidences = []
        
        for cat_name, metrics in performance.items():
            if metrics['count'] > 0:
                categories.append(cat_name.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
                f1_scores.append(metrics['f1'])
                counts.append(metrics['count'])
                confidences.append(metrics['avg_confidence'])
        
        # Create subplots - now with 2x3 layout to include confusion matrix
        fig = plt.figure(figsize=(18, 12))
        
        # Accuracy comparison
        ax1 = plt.subplot(2, 3, 1)
        bars1 = ax1.bar(categories, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy by User Category')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add aggregate line
        ax1.axhline(y=aggregate_metrics['overall_accuracy'], color='red', linestyle='--', 
                   label=f'Overall: {aggregate_metrics["overall_accuracy"]:.3f}')
        ax1.legend()
        
        # Add count labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # F1 Score comparison
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(categories, f1_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Model F1 Score by User Category')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add aggregate line
        ax2.axhline(y=aggregate_metrics['overall_f1'], color='red', linestyle='--', 
                   label=f'Overall: {aggregate_metrics["overall_f1"]:.3f}')
        ax2.legend()
        
        # Confidence comparison
        ax3 = plt.subplot(2, 3, 3)
        bars3 = ax3.bar(categories, confidences, color='orange', alpha=0.7)
        ax3.set_title('Average Confidence by User Category')
        ax3.set_ylabel('Average Confidence')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add aggregate line
        ax3.axhline(y=aggregate_metrics['average_confidence'], color='red', linestyle='--',
                   label=f'Overall: {aggregate_metrics["average_confidence"]:.3f}')
        ax3.legend()
        
        # Sample size comparison
        ax4 = plt.subplot(2, 3, 4)
        bars4 = ax4.bar(categories, counts, color='lightcoral', alpha=0.7)
        ax4.set_title('Sample Size by User Category')
        ax4.set_ylabel('Number of Conversations')
        ax4.tick_params(axis='x', rotation=45)
        
        # Promise confusion matrix (simplified for overview)
        if promise_cm is not None and promise_metrics is not None:
            ax5 = plt.subplot(2, 3, 5)
            sns.heatmap(promise_cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                       xticklabels=['Pred: Broken', 'Pred: Kept'],
                       yticklabels=['Act: Broken', 'Act: Kept'])
            ax5.set_title(f'Promise Matrix\n(Acc: {promise_metrics["promise_accuracy"]:.3f})')
            
            # Aggregate metrics summary
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            summary_text = f"""AGGREGATE METRICS
            
Total Conversations: {aggregate_metrics['total_conversations']}
Overall Accuracy: {aggregate_metrics['overall_accuracy']:.3f}
Overall F1 (Weighted): {aggregate_metrics['overall_f1']:.3f}
Overall F1 (Macro): {aggregate_metrics['overall_macro_f1']:.3f}
Overall F1 (Micro): {aggregate_metrics['overall_micro_f1']:.3f}

PROMISE ANALYSIS
Total Promise Conversations: {promise_metrics['total_promise_conversations']}
Promise Prediction Accuracy: {promise_metrics['promise_accuracy']:.3f}
Promise F1 Score: {promise_metrics['promise_f1']:.3f}
Promise Recall: {promise_metrics['promise_recall']:.3f}
Promise Precision: {promise_metrics['promise_precision']:.3f}"""
            
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('model_performance_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed standalone promise confusion matrix plot
        if promise_cm is not None and promise_metrics is not None:
            self.plot_promise_confusion_matrix(promise_cm, promise_metrics)
    
    def generate_detailed_report(self) -> str:
        """
        Generate a comprehensive report comparing model performance
        """
        categories = self.categorize_users()
        performance = self.calculate_performance_by_category(categories)
        timing_analysis = self.analyze_decision_timing_impact(categories)
        aggregate_metrics = self.calculate_aggregate_metrics()
        promise_cm, promise_metrics = self.create_promise_confusion_matrix(categories)
        
        report = []
        report.append("="*80)
        report.append("MODEL PERFORMANCE vs PERSUASION BEHAVIOR ANALYSIS")
        report.append("="*80)
        
        # Aggregate metrics first
        report.append(f"\nAGGREGATE MODEL PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Total Conversations: {aggregate_metrics['total_conversations']}")
        report.append(f"Overall Accuracy: {aggregate_metrics['overall_accuracy']:.4f}")
        report.append(f"Overall F1 (Weighted): {aggregate_metrics['overall_f1']:.4f}")
        report.append(f"Overall F1 (Macro): {aggregate_metrics['overall_macro_f1']:.4f}")
        report.append(f"Overall F1 (Micro): {aggregate_metrics['overall_micro_f1']:.4f}")
        report.append(f"Overall Precision: {aggregate_metrics['overall_precision']:.4f}")
        report.append(f"Overall Recall: {aggregate_metrics['overall_recall']:.4f}")
        report.append(f"Average Confidence: {aggregate_metrics['average_confidence']:.4f}")
        report.append(f"Correct Predictions: {aggregate_metrics['correct_predictions']}")
        
        # Promise-specific analysis
        if promise_metrics and 'error' not in promise_metrics:
            report.append(f"\nPROMISE-KEEPING PREDICTION ANALYSIS:")
            report.append("-" * 40)
            report.append(f"Promise Conversations Analyzed: {promise_metrics['total_promise_conversations']}")
            report.append(f"Promise Prediction Accuracy: {promise_metrics['promise_accuracy']:.4f}")
            report.append(f"Promise F1 Score: {promise_metrics['promise_f1']:.4f}")
            report.append(f"Promise Precision: {promise_metrics['promise_precision']:.4f}")
            report.append(f"Promise Recall: {promise_metrics['promise_recall']:.4f}")
            
            if promise_cm is not None:
                report.append(f"\nPromise Confusion Matrix:")
                report.append(f"                    Predicted")
                report.append(f"                Broken    Kept")
                report.append(f"Actual Broken   {promise_cm[0,0]:6d}  {promise_cm[0,1]:6d}")
                report.append(f"Actual Kept     {promise_cm[1,0]:6d}  {promise_cm[1,1]:6d}")
                
                # Calculate specific error types
                true_negatives = promise_cm[0,0]  # Correctly predicted promise broken
                false_positives = promise_cm[0,1]  # Predicted kept but actually broken
                false_negatives = promise_cm[1,0]  # Predicted broken but actually kept
                true_positives = promise_cm[1,1]   # Correctly predicted promise kept
                
                report.append(f"\nPromise Prediction Breakdown:")
                report.append(f"  • Correctly predicted promise kept: {true_positives}")
                report.append(f"  • Correctly predicted promise broken: {true_negatives}")
                report.append(f"  • Falsely predicted kept (actually broken): {false_positives}")
                report.append(f"  • Falsely predicted broken (actually kept): {false_negatives}")
                
                if false_positives > false_negatives:
                    report.append(f"  → Model tends to be overly optimistic about promise-keeping")
                elif false_negatives > false_positives:
                    report.append(f"  → Model tends to be overly pessimistic about promise-keeping")
        
        # Category breakdown
        total_conversations = len(self.merged_data)
        report.append(f"\nUSER CATEGORY BREAKDOWN:")
        report.append("-" * 40)
        
        for cat_name, cat_df in categories.items():
            count = len(cat_df)
            percentage = (count / total_conversations) * 100 if total_conversations > 0 else 0
            report.append(f"{cat_name.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Performance analysis
        report.append(f"\nMODEL PERFORMANCE BY CATEGORY:")
        report.append("-" * 40)
        
        # Focus on promise keepers vs breakers
        if performance['promise_keepers']['count'] > 0 and performance['promise_breakers']['count'] > 0:
            pk_acc = performance['promise_keepers']['accuracy']
            pb_acc = performance['promise_breakers']['accuracy']
            pk_conf = performance['promise_keepers']['avg_confidence']
            pb_conf = performance['promise_breakers']['avg_confidence']
            pk_f1 = performance['promise_keepers']['f1']
            pb_f1 = performance['promise_breakers']['f1']
            
            report.append(f"\n🎯 KEY FINDING - PROMISE ANALYSIS:")
            report.append(f"Promise Keepers (said yes, donated):")
            report.append(f"  • Accuracy: {pk_acc:.4f} | F1: {pk_f1:.4f} | Confidence: {pk_conf:.4f}")
            report.append(f"  • Sample size: {performance['promise_keepers']['count']}")
            
            report.append(f"\nPromise Breakers (said yes, didn't donate):")
            report.append(f"  • Accuracy: {pb_acc:.4f} | F1: {pb_f1:.4f} | Confidence: {pb_conf:.4f}")
            report.append(f"  • Sample size: {performance['promise_breakers']['count']}")
            
            acc_diff = pk_acc - pb_acc
            f1_diff = pk_f1 - pb_f1
            conf_diff = pk_conf - pb_conf
            
            report.append(f"\n📊 Performance Differences:")
            report.append(f"  • Accuracy difference: {acc_diff:+.4f} (keepers vs breakers)")
            report.append(f"  • F1 difference: {f1_diff:+.4f} (keepers vs breakers)")
            report.append(f"  • Confidence difference: {conf_diff:+.4f} (keepers vs breakers)")
            
            if acc_diff > 0.05:
                report.append("   → Model performs significantly better on promise keepers")
                report.append("   → Suggests difficulty detecting deceptive behavior")
            elif acc_diff < -0.05:
                report.append("   → Model performs significantly better on promise breakers")
                report.append("   → Model may be good at detecting inconsistency signals")
            else:
                report.append("   → Similar performance on both groups")
        
        # Detailed performance for all categories
        report.append(f"\nDETAILED PERFORMANCE METRICS:")
        report.append("-" * 40)
        
        for cat_name, metrics in performance.items():
            if metrics['count'] > 0:
                report.append(f"\n{cat_name.replace('_', ' ').upper()}:")
                report.append(f"  Sample Size: {metrics['count']}")
                report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
                report.append(f"  Precision: {metrics['precision']:.4f}")
                report.append(f"  Recall: {metrics['recall']:.4f}")
                report.append(f"  F1 Score: {metrics['f1']:.4f}")
                report.append(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
                report.append(f"  Correct Predictions: {metrics['correct_predictions']}")
        
        # Decision timing analysis
        if timing_analysis:
            report.append(f"\nDECISION TIMING IMPACT:")
            report.append("-" * 40)
            
            for cat_name, timing_data in timing_analysis.items():
                if timing_data:
                    report.append(f"\n{cat_name.replace('_', ' ').title()}:")
                    for timing_bin, timing_metrics in timing_data.items():
                        report.append(f"  {timing_bin}: {timing_metrics['accuracy']:.4f} accuracy "
                                    f"({timing_metrics['count']} samples)")
        
        # Statistical insights
        report.append(f"\nSTATISTICAL INSIGHTS:")
        report.append("-" * 40)
        
        # Calculate correlation between consistency and model performance
        consistency_scores = []
        accuracy_scores = []
        
        for cat_name, metrics in performance.items():
            if metrics['count'] > 5:  # Only consider categories with sufficient samples
                consistency_scores.append(1 if 'keeper' in cat_name or 'consistent' in cat_name else 0)
                accuracy_scores.append(metrics['accuracy'])
        
        if len(consistency_scores) > 1:
            correlation = np.corrcoef(consistency_scores, accuracy_scores)[0, 1]
            report.append(f"Correlation between user consistency and model accuracy: {correlation:.4f}")
            
            if correlation > 0.3:
                report.append("→ Model performs better on consistent users")
            elif correlation < -0.3:
                report.append("→ Model performs better on inconsistent users")
            else:
                report.append("→ No strong correlation found")
        
        return "\n".join(report)
    
    def save_analysis(self, output_prefix: str = 'model_persuasion_analysis'):
        """
        Save all analysis results
        """
        categories = self.categorize_users()
        performance = self.calculate_performance_by_category(categories)
        timing_analysis = self.analyze_decision_timing_impact(categories)
        aggregate_metrics = self.calculate_aggregate_metrics()
        promise_cm, promise_metrics = self.create_promise_confusion_matrix(categories)
        
        # Save detailed results
        analysis_results = {
            'aggregate_metrics': aggregate_metrics,
            'promise_analysis': {
                'confusion_matrix': promise_cm.tolist() if promise_cm is not None else None,
                'metrics': promise_metrics
            },
            'categories': {k: v.to_dict('records') if len(v) > 0 else [] for k, v in categories.items()},
            'performance_metrics': performance,
            'timing_analysis': timing_analysis,
            'merged_data_summary': {
                'total_conversations': len(self.merged_data),
                'unique_labels': self.merged_data['label'].unique().tolist(),
                'overall_accuracy': accuracy_score(self.merged_data['label'], self.merged_data['predicted_label'])
            }
        }
        
        with open(f'{output_prefix}_detailed.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save report
        report = self.generate_detailed_report()
        with open(f'{output_prefix}_report.txt', 'w') as f:
            f.write(report)
        
        # Create visualization
        self.create_performance_comparison_plot(performance, aggregate_metrics, promise_cm, promise_metrics)
        
        print(f"Analysis saved to:")
        print(f"  • {output_prefix}_detailed.json")
        print(f"  • {output_prefix}_report.txt") 
        print(f"  • model_performance_comprehensive.png")
        print(f"  • promise_confusion_matrix.png")
        
        return report

# Example usage
def main():
    # File paths
    JSON_FILE = "baselines/persuasion_analysis_results.json"  # Your persuasion analysis file
    CSV_FILE = "src/sft/results/p4g/seed_11611/p4g_classification_ratio_1_none_predictions.csv"  # Your model results file
    
    # Initialize comparison
    analyzer = ModelPersuasionComparison(JSON_FILE, CSV_FILE)
    
    # Generate and display report
    report = analyzer.save_analysis()
    
    print(report)
    
    # Additional specific analysis for promise behavior
    categories = analyzer.categorize_users()
    aggregate_metrics = analyzer.calculate_aggregate_metrics()
    promise_cm, promise_metrics = analyzer.create_promise_confusion_matrix(categories)
    
    print(f"\n" + "="*60)
    print("QUICK SUMMARY:")
    print("="*60)
    
    print(f"\nAGGREGATE METRICS:")
    print(f"Overall Accuracy: {aggregate_metrics['overall_accuracy']:.4f}")
    print(f"Overall F1 (Weighted): {aggregate_metrics['overall_f1']:.4f}")
    print(f"Overall F1 (Macro): {aggregate_metrics['overall_macro_f1']:.4f}")
    
    print(f"\nUSER DISTRIBUTION:")
    print(f"Promise Keepers: {len(categories['promise_keepers'])}")
    print(f"Promise Breakers: {len(categories['promise_breakers'])}")
    print(f"Refusal Reversers: {len(categories['refusal_reversers'])}")
    print(f"Consistent Refusers: {len(categories['consistent_refusers'])}")
    
    if promise_metrics and 'error' not in promise_metrics:
        print(f"\nPROMISE PREDICTION PERFORMANCE:")
        print(f"Promise Accuracy: {promise_metrics['promise_accuracy']:.4f}")
        print(f"Promise F1 Score: {promise_metrics['promise_f1']:.4f}")
        print(f"Promise Precision: {promise_metrics['promise_precision']:.4f}")
        print(f"Promise Recall: {promise_metrics['promise_recall']:.4f}")
        
        if promise_cm is not None:
            print(f"\nCONFUSION MATRIX (Promise Kept/Broken):")
            print(f"                Predicted")
            print(f"            Broken   Kept")
            print(f"Act. Broken  {promise_cm[0,0]:3d}    {promise_cm[0,1]:3d}")
            print(f"Act. Kept    {promise_cm[1,0]:3d}    {promise_cm[1,1]:3d}")

if __name__ == "__main__":
    main()