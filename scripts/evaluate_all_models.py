import os
import yaml
import json
import argparse
from evaluate import main as evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_comparison_plot(all_metrics, output_path):
    """Create a comparison plot of all models"""
    # Extract metrics for plotting
    models = []
    accuracies = []
    f1_scores = []
    
    for model_name, metrics in all_metrics.items():
        models.append(model_name)
        accuracies.append(metrics.get('accuracy', 0))
        f1_scores.append(metrics.get('f1_score', 0))
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Accuracy plot
    ax1.bar(models, accuracies, color='skyblue')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # F1 Score plot
    ax2.bar(models, f1_scores, color='lightcoral')
    ax2.set_title('Model F1 Score Comparison')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Evaluate all trained models"""
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Get list of models
    models = list(params['models'].keys())
    print(f"Evaluating {len(models)} models: {models}")
    
    # Create reports directory
    os.makedirs('reports/metrics', exist_ok=True)
    os.makedirs('reports/plots', exist_ok=True)
    
    all_metrics = {}
    
    # Evaluate each model
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print('='*50)
        
        # Check if model exists
        model_path = f"models/{model_name}/model.pkl"
        if not os.path.exists(model_path):
            print(f"✗ Model {model_name} not found at {model_path}")
            continue
            
        try:
            # Create args for evaluate.py
            args = argparse.Namespace(model=model_name)
            evaluate_model(args)
            
            # Load the metrics
            metrics_file = f"reports/metrics/{model_name}_evaluation.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    all_metrics[model_name] = metrics
                    print(f"✓ {model_name} evaluated successfully")
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
    
    # Create summary report
    summary = {
        'models_evaluated': len(all_metrics),
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'model_metrics': all_metrics,
        'best_model': max(all_metrics.items(), key=lambda x: x[1].get('accuracy', 0))[0] if all_metrics else None
    }
    
    with open('reports/metrics/evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison plot
    if all_metrics:
        create_comparison_plot(all_metrics, 'reports/plots/model_comparison.png')
    
    print("\nAll models evaluated!")
    print(f"Summary saved to: reports/metrics/evaluation_summary.json")
    print(f"Comparison plot saved to: reports/plots/model_comparison.png")

if __name__ == "__main__":
    main()