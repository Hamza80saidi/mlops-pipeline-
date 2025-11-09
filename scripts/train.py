import os
import sys
import argparse
import yaml
import json
import pandas as pd
import numpy as np
import joblib
import mlflow 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.strategies.model_factory import ModelFactory
from src.ml.strategies.training_strategy import StandardTrainingStrategy, TransferLearningStrategy
from src.ml.preprocessing.text_preprocessor import TextPreprocessor
from src.core.config import settings

# Set environment variables for MLflow and MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True, 
                          choices=['svm', 'random_forest', 'logistic_regression', 'albert'])
        args = parser.parse_args()
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Load data
    train_df = pd.read_csv(params['data']['processed_train'])
    
    # Get number of unique labels (for ALBERT)
    num_labels = len(train_df['label_encoded'].unique())
    
    # Create output directory
    model_dir = f"models/{args.model}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize text preprocessor
    text_preprocessor = TextPreprocessor(
        lowercase=params['preprocessing']['lowercase'],
        remove_punctuation=params['preprocessing']['remove_punctuation']
    )
    
    # Prepare features
    if args.model != 'albert':
        # Preprocess text
        processed_texts = [text_preprocessor.preprocess(text) for text in train_df['text'].values]
        
        # Use TF-IDF for traditional ML models
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(processed_texts).toarray()
        
        # Save vectorizer
        joblib.dump(vectorizer, f"{model_dir}/vectorizer.pkl")
    else:
        # ALBERT handles its own text processing
        X_train = train_df['text'].values
    
    y_train = train_df['label_encoded'].values
    
    # Split into train/validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, 
        test_size=params['training']['validation_split'],
        random_state=params['data']['random_state']
    )
    
    # Create model
    model_config = params['models'][args.model]
    model = ModelFactory.create_model(args.model, model_config)
    
    # For ALBERT, prepare with number of labels
    if args.model == 'albert':
        model.prepare_for_training(num_labels)
    
    # Set up MLflow (optional - can skip if not running)
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(params['mlflow']['experiment_name'])
        use_mlflow = True
    except Exception as e:
        print(f"Warning: MLflow not available: {e}")
        use_mlflow = False
    
    print(f"Training {args.model} model...")
    print(f"Number of labels: {num_labels}")
    print(f"Training samples: {len(X_train_split)}")
    print(f"Validation samples: {len(X_val_split)}")
    
    if use_mlflow:
        with mlflow.start_run(run_name=f"{args.model}_training"):
            # Log parameters
            mlflow.log_params(model_config)
            mlflow.log_params({
                'model_type': args.model,
                'training_samples': len(X_train_split),
                'validation_samples': len(X_val_split),
                'num_labels': num_labels
            })
            
            # Train model
            if args.model == 'albert':
                # ALBERT returns metrics directly
                metrics = model.train(X_train_split, y_train_split, X_val_split, y_val_split)
            else:
                # Traditional ML models use strategy pattern
                strategy = StandardTrainingStrategy()
                model, metrics = strategy.execute(model, X_train_split, y_train_split, X_val_split, y_val_split)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save model
            model_path = f"{model_dir}/model.pkl"
            model.save_model(model_path)
            
            # Save metrics locally
            with open(f'{model_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Log model to MLflow
            if args.model != 'albert':
                mlflow.sklearn.log_model(model.model, args.model)
    else:
        # Train without MLflow
        if args.model == 'albert':
            metrics = model.train(X_train_split, y_train_split, X_val_split, y_val_split)
        else:
            strategy = StandardTrainingStrategy()
            model, metrics = strategy.execute(model, X_train_split, y_train_split, X_val_split, y_val_split)
        
        # Save model
        model_path = f"{model_dir}/model.pkl"
        model.save_model(model_path)
        
        # Save metrics locally
        with open(f'{model_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"Model {args.model} trained successfully!")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()