import os
import yaml
import argparse
from train import main as train_model

def main():
    """Train all models defined in params.yaml"""
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Get list of models
    models = list(params['models'].keys())
    print(f"Training {len(models)} models: {models}")
    
    # Train each model
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print('='*50)
        
        try:
            # Create args for train.py
            args = argparse.Namespace(model=model_name)
            train_model(args)
            print(f"✓ {model_name} trained successfully")
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
    
    print("\nAll models trained!")

if __name__ == "__main__":
    main()