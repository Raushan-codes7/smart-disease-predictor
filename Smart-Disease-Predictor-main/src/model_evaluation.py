"""
Model evaluation module for disease prediction system.
Handles evaluation metrics, confusion matrix, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pickle
import os


def evaluate_model(y_true, y_pred, y_pred_proba, label_encoder, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Prediction probabilities
        label_encoder: Label encoder for disease names
        model_name (str): Name of the model for display
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    print(f"Evaluating {model_name}...")
    print("=" * 50)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Store all metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'model_name': model_name
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path=None, figsize=(12, 10)):
    """
    Create and display confusion matrix heatmap.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        label_encoder: Label encoder for disease names
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    # Get class names
    class_names = label_encoder.classes_
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Disease Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Disease', fontsize=12)
    plt.ylabel('Actual Disease', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, y_pred_proba, label_encoder, save_path=None, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Prediction probabilities
        label_encoder: Label encoder for disease names
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    # Binarize labels for multi-class ROC
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=figsize)
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=2)
    
    # Plot ROC curves for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Formatting
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class Disease Prediction', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_class_distribution(y_true, label_encoder, save_path=None, figsize=(12, 6)):
    """
    Plot distribution of classes in the dataset.
    
    Args:
        y_true (np.ndarray): True labels
        label_encoder: Label encoder for disease names
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    class_names = label_encoder.classes_
    class_counts = np.bincount(y_true)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(class_names)), class_counts, color='skyblue', edgecolor='navy')
    
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Disease Classes in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None, figsize=(10, 6)):
    """
    Plot comparison of different metrics.
    
    Args:
        metrics_dict (dict): Dictionary of metrics from different models
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    # Extract metrics
    models = list(metrics_dict.keys())
    accuracy = [metrics_dict[model]['accuracy'] for model in models]
    precision = [metrics_dict[model]['precision_weighted'] for model in models]
    recall = [metrics_dict[model]['recall_weighted'] for model in models]
    f1 = [metrics_dict[model]['f1_weighted'] for model in models]
    
    # Create plot
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='lightcoral')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='lightgreen')
    bars4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='gold')
    
    # Formatting
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    plt.show()


def save_evaluation_results(metrics, output_dir='results'):
    """
    Save evaluation results to files.
    
    Args:
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as pickle
    metrics_path = f'{output_dir}/evaluation_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame([{
        'Model': metrics['model_name'],
        'Accuracy': metrics['accuracy'],
        'Precision (Macro)': metrics['precision_macro'],
        'Precision (Weighted)': metrics['precision_weighted'],
        'Recall (Macro)': metrics['recall_macro'],
        'Recall (Weighted)': metrics['recall_weighted'],
        'F1-Score (Macro)': metrics['f1_macro'],
        'F1-Score (Weighted)': metrics['f1_weighted']
    }])
    
    csv_path = f'{output_dir}/evaluation_summary.csv'
    metrics_df.to_csv(csv_path, index=False)
    
    print(f"Evaluation results saved to {output_dir}/")
    print(f"- Metrics: {metrics_path}")
    print(f"- Summary: {csv_path}")


def generate_evaluation_report(y_true, y_pred, y_pred_proba, label_encoder, 
                             model_name="Disease Predictor", save_dir='results'):
    """
    Generate comprehensive evaluation report with all visualizations.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Prediction probabilities
        label_encoder: Label encoder for disease names
        model_name (str): Name of the model
        save_dir (str): Directory to save all results
    """
    print(f"Generating comprehensive evaluation report for {model_name}...")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = evaluate_model(y_true, y_pred, y_pred_proba, label_encoder, model_name)
    
    # Create visualizations
    plot_confusion_matrix(y_true, y_pred, label_encoder, 
                         save_path=f'{save_dir}/confusion_matrix.png')
    
    plot_roc_curves(y_true, y_pred_proba, label_encoder, 
                   save_path=f'{save_dir}/roc_curves.png')
    
    plot_class_distribution(y_true, label_encoder, 
                           save_path=f'{save_dir}/class_distribution.png')
    
    # Save results
    save_evaluation_results(metrics, save_dir)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print("-" * 40)
    class_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    print(f"\nEvaluation report generated successfully!")
    print(f"All files saved to: {save_dir}/")
    
    return metrics


if __name__ == "__main__":
    # Test the evaluation pipeline
    from data_preprocessing import load_preprocessed_data
    from model_training import load_model
    
    # Load data and model
    X_train, X_test, y_train, y_test, label_encoder, feature_names = load_preprocessed_data()
    model, _, _, _ = load_model()
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Generate evaluation report
    metrics = generate_evaluation_report(y_test, y_pred, y_pred_proba, label_encoder)
