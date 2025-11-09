import os
import sys
import yaml
import json
import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.models.albert_model import ALBERTModel

def test_albert_training():
    print("Testing ALBERT model training...")
    
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    
    # Get a small sample for testing
    sample_size = 10  # Very small for quick test
    sample_df = train_df.head(sample_size)
    
    X_train = sample_df['text'].values
    y_train = sample_df['label_encoded'].values
    
    # Get number of unique labels
    num_labels = len(train_df['label_encoded'].unique())
    print(f"Number of labels: {num_labels}")
    
    # Create minimal config for testing
    config = {
        'model_name': 'albert-base-v2',
        'max_length': 64,  # Very short for testing
        'batch_size': 2,   # Very small batch
        'epochs': 1,       # Single epoch
        'learning_rate': 2e-5
    }
    
    # Create and train model
    model = ALBERTModel(config)
    
    # ALBERT needs to know the number of labels
    model.prepare_for_training(num_labels)
    
    try:
        print(f"Training on {len(X_train)} samples...")
        metrics = model.train(X_train, y_train)
        print(f"Training completed! Metrics: {metrics}")
        
        # Save the model
        os.makedirs('models/albert', exist_ok=True)
        model.save_model('models/albert/model.pkl')
        
        # Save metrics
        with open('models/albert/metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        print("ALBERT model saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_albert_training()