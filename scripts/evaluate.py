import os
import sys
import argparse
import yaml
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.strategies.model_factory import ModelFactory
from src.ml.preprocessing.text_preprocessor import TextPreprocessor

def plot_confusion_matrix(cm, labels, save_path):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_classification_report(report_dict, save_path):
    """Create and save classification report plot"""
    # Convert to dataframe for plotting
    df = pd.DataFrame(report_dict).transpose()
    df = df[['precision', 'recall', 'f1-score']].iloc[:-3]  # Remove avg rows
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax)
    ax.set_title('Classification Report')
    ax.set_ylabel('Score')
    ax.set_xlabel('Class')
    ax.legend(['Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True,
                          choices=['svm', 'random_forest', 'logistic_regression', 'albert'])
        args = parser.parse_args()
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Load test data
    test_df = pd.read_csv(params['data']['processed_test'])
    
    # Load label mapping
    with open('data/processed/label_mapping.yaml', 'r') as f:
        label_mapping = yaml.safe_load(f)
    labels = list(label_mapping.keys())
    
    # Create output directories
    os.makedirs('reports/metrics', exist_ok=True)
    os.makedirs('reports/plots', exist_ok=True)
    
    # Initialize text preprocessor
    text_preprocessor = TextPreprocessor(
        lowercase=params['preprocessing']['lowercase'],
        remove_punctuation=params['preprocessing']['remove_punctuation']
    )
    
    # Prepare features
    if args.model != 'albert':
        # Load vectorizer
        vectorizer_path = f"models/{args.model}/vectorizer.pkl"
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
        
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess text
        processed_texts = [text_preprocessor.preprocess(text) for text in test_df['text'].values]
        X_test = vectorizer.transform(processed_texts).toarray()
    else:
        # ALBERT handles its own text processing
        X_test = test_df['text'].values
    
    y_test = test_df['label_encoded'].values
    
    # Load model
    model_path = f"models/{args.model}/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_config = params['models'][args.model]
    model = ModelFactory.create_model(args.model, model_config)
    model.load_model(model_path)
    
    print(f"Evaluating {args.model} model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    
    # Add to metrics
    metrics['classification_report'] = report
    metrics['confusion_matrix'] = cm.tolist()
    
    # Save metrics
    metrics_file = f'reports/metrics/{args.model}_evaluation.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots
    plot_confusion_matrix(cm, labels, f'reports/plots/{args.model}_confusion_matrix.png')
    plot_classification_report(report, f'reports/plots/{args.model}_classification_report.png')
    
    print(f"Evaluation complete for {args.model}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Results saved to: {metrics_file}")

if __name__ == "__main__":
    main()