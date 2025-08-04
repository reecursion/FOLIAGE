import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import argparse
from typing import List, Tuple
import os

def get_class_names():
    """Return the class names mapping"""
    return [
        'expressed_donate_did',      # 0: Expressed intention of donating but did not donate
        'expressed_donate_donated',  # 1: Expressed intention and donated  
        'no_express_donated',        # 2: Did not express intention but donated
        'no_express_no_donate',      # 3: Did not express intention and did not donate
        'unclear_donated',           # 4: Unclear but donated
        'unclear_no_donate'          # 5: Unclear but did not donate
    ]

def get_short_class_names():
    """Return shortened class names for better visualization"""
    return [
        'Promised→No Donate',      # 0: Expressed intention of donating but did not donate
        'Promised→Donated',  # 1: Expressed intention and donated  
        'Refused→Donated', # 2: Did not express intention but donated
        'Refused→No Donate',    # 3: Did not express intention and did not donate
        'Unclear→Donated',  # 4: Unclear but donated
        'Unclear→No Donate'       # 5: Unclear but did not donate
    ]

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], 
                         title: str = "Confusion Matrix",
                         save_path: str = None,
                         figsize: Tuple[int, int] = (12, 10),
                         use_short_names: bool = True):
    """
    Plot a detailed confusion matrix with percentages and counts
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
        save_path: Path to save the plot
        figsize: Figure size
        use_short_names: Whether to use shortened class names
    """
    
    # Use short names if requested
    display_names = get_short_class_names() if use_short_names else class_names
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Confusion matrix with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=display_names, yticklabels=display_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'{title} - Counts')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot 2: Confusion matrix with percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=display_names, yticklabels=display_names,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title(f'{title} - Percentages')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, cm_percent

def plot_single_confusion_matrix(y_true: List[int], y_pred: List[int], 
                                class_names: List[str], 
                                title: str = "Confusion Matrix",
                                save_path: str = None,
                                figsize: Tuple[int, int] = (10, 8),
                                use_short_names: bool = True,
                                show_percentages: bool = True):
    """
    Plot a single confusion matrix with both counts and percentages
    """
    
    # Use short names if requested
    display_names = get_short_class_names() if use_short_names else class_names
    labels = list(range(len(class_names)))  # [0, 1, 2, 3, 4, 5]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    if show_percentages:
        annotations = np.array([[f'{count}\n({percent:.1f}%)' 
                               for count, percent in zip(row_counts, row_percents)]
                               for row_counts, row_percents in zip(cm, cm_percent)])
    else:
        annotations = cm
    
    # Create the plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, cm_percent

def analyze_performance(y_true: List[int], y_pred: List[int], class_names: List[str]):
    """
    Provide detailed performance analysis
    """
    
    print("="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\nClassification Report:")
    print("-" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 zero_division=0, digits=4)
    print(report)
    
    # Per-class analysis
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nPer-Class Analysis:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            true_negatives = cm.sum() - true_positives - false_positives - false_negatives
            
            total_true = cm[i, :].sum()
            total_pred = cm[:, i].sum()
            
            precision = true_positives / total_pred if total_pred > 0 else 0
            recall = true_positives / total_true if total_true > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  True instances: {total_true}")
            print(f"  Predicted instances: {total_pred}")
            print(f"  Correct predictions: {true_positives}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
    
    # Most confused pairs
    print(f"\nMost Confused Class Pairs:")
    print("-" * 50)
    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], class_names[i], class_names[j]))
    
    # Sort by confusion count
    confusion_pairs.sort(reverse=True)
    
    for count, true_class, pred_class in confusion_pairs[:10]:  # Top 10
        print(f"  {true_class} → {pred_class}: {count} cases")

def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix from predictions with ground truth')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file with predictions and ground truth')
    parser.add_argument('--output_dir', type=str, default='./plots/',
                       help='Directory to save plots')
    parser.add_argument('--title', type=str, default='p4g - Face-Saving Strategies',
                       help='Title for the confusion matrix')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 8],
                       help='Figure size (width height)')
    parser.add_argument('--use_short_names', action='store_true', default=True,
                       help='Use shortened class names for better readability')
    parser.add_argument('--plot_type', type=str, choices=['single', 'double'], default='single',
                       help='Plot type: single (counts+percentages) or double (separate plots)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(args.csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Filter to only cases with ground truth
    df_with_gt = df[df['has_ground_truth'] == True].copy()
    print(f"Found {len(df_with_gt)} cases with ground truth")
    
    if len(df_with_gt) == 0:
        print("Error: No cases with ground truth found!")
        return
    
    # Extract true and predicted labels
    y_true = df_with_gt['ground_truth_label'].values
    y_pred = df_with_gt['final_prediction'].values
    
    # Get class names
    class_names = get_class_names()
    
    # Plot confusion matrix
    if args.plot_type == 'double':
        save_path = os.path.join(args.output_dir, 'confusion_matrix_double.png')
        cm, cm_percent = plot_confusion_matrix(
            y_true, y_pred, class_names, 
            title=args.title,
            save_path=save_path,
            figsize=tuple(args.figsize),
            use_short_names=args.use_short_names
        )
    else:
        save_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        cm, cm_percent = plot_single_confusion_matrix(
            y_true, y_pred, class_names,
            title=args.title,
            save_path=save_path,
            figsize=tuple(args.figsize),
            use_short_names=args.use_short_names
        )
    
    # Perform detailed analysis
    analyze_performance(y_true, y_pred, class_names)
    
    # Save analysis to file
    analysis_file = os.path.join(args.output_dir, 'performance_analysis.txt')
    
    # Redirect print output to file
    import sys
    original_stdout = sys.stdout
    
    with open(analysis_file, 'w') as f:
        sys.stdout = f
        analyze_performance(y_true, y_pred, class_names)
        
        # Add confusion matrix values
        print(f"\n\nConfusion Matrix (Counts):")
        print("-" * 50)
        print("True\\Pred", end="")
        for name in (get_short_class_names() if args.use_short_names else class_names):
            print(f"\t{name[:8]}", end="")
        print()
        
        for i, (true_name, row) in enumerate(zip(class_names, cm)):
            display_name = get_short_class_names()[i] if args.use_short_names else true_name
            print(f"{display_name[:8]}", end="")
            for val in row:
                print(f"\t{val}", end="")
            print()
    
    sys.stdout = original_stdout
    
    print(f"\nPerformance analysis saved to: {analysis_file}")
    
    # Create a summary CSV
    summary_data = []
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            total_true = cm[i, :].sum()
            total_pred = cm[:, i].sum()
            correct = cm[i, i]
            
            precision = correct / total_pred if total_pred > 0 else 0
            recall = correct / total_true if total_true > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            summary_data.append({
                'class': class_name,
                'true_instances': total_true,
                'predicted_instances': total_pred,
                'correct_predictions': correct,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(args.output_dir, 'class_performance_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Class performance summary saved to: {summary_file}")

if __name__ == '__main__':
    main()