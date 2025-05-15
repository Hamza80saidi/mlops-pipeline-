import argparse
import yaml
import json
import mlflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.ml.strategies.model_factory import ModelFactory
from src.ml.strategies.training_strategy import StandardTrainingStrategy, TransferLearningStrategy
from src.ml.evaluation.metrics import calculate_metrics
from src.core.config import settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Load data
    train_df = pd.read_csv(params['data']['processed_train'])
    
    # Prepare features
    if args.model != 'albert':
        # Use TF-IDF for traditional ML models
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_df['text']).toarray()
    else:
        # ALBERT handles its own text processing
        X_train = train_df['text'].values
    
    y_train = train_df['label_encoded'].values
    
    # Create model
    model_config = params['models'][args.model]
    model = ModelFactory.create_model(args.model, model_config)
    
    # Select training strategy
    if args.model == 'albert':
        strategy = TransferLearningStrategy()
    else:
        strategy = StandardTrainingStrategy()
    
    # Set up MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=f"{args.model}_training"):
        # Log parameters
        mlflow.log_params(model_config)
        mlflow.log_params(params['training'])
        
        # Train model
        model, metrics = strategy.execute(model, X_train, y_train)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save model
        model_path = f"models/{args.model}/model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        
        # Log model to MLflow
        if args.model != 'albert':
            mlflow.sklearn.log_model(model.model, args.model)
        else:
            mlflow.pytorch.log_model(model.model, args.model)
        
        # Save metrics locally
        with open(f'models/{args.model}/metrics.json', 'w') as f:
            json.dump(metrics, f)
    
    print(f"Model {args.model} trained successfully!")

if __name__ == "__main__":
    main()