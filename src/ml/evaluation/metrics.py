from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      labels: List[str] = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Per-class metrics
    if labels:
        per_class_metrics = classification_report(
            y_true, y_pred, 
            target_names=labels, 
            output_dict=True
        )
        metrics['per_class'] = per_class_metrics
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                          save_path: str = None) -> None:
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                            save_path: str = None) -> None:
    """Plot comparison of metrics across models"""
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()