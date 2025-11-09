import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
import os

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(params['data']['raw_data'])
    
    # Basic cleaning
    df = df.dropna()
    df['text'] = df['text'].str.strip()
    
    # Debug: Check what columns exist
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find the label column (could be 'label' or 'disease' or similar)
    label_column = None
    for col in ['label', 'disease', 'diagnosis', 'condition', 'category']:
        if col in df.columns:
            label_column = col
            break
    
    if label_column is None:
        # If no label column found, check if the first non-text column is the label
        for col in df.columns:
            if col != 'text' and df[col].dtype == 'object':
                label_column = col
                break
    
    if label_column is None:
        raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
    
    print(f"Using '{label_column}' as label column")
    
    # Rename to standard 'label' column
    if label_column != 'label':
        df['label'] = df[label_column]
    
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # Create label mapping with actual disease names
    label_mapping = {}
    for i, label in enumerate(le.classes_):
        label_mapping[str(label)] = int(i)  # Ensure string keys and int values
    
    # Save label mapping
    with open('data/processed/label_mapping.yaml', 'w') as f:
        yaml.dump(label_mapping, f)
    
    print("\nLabel mapping created:")
    for disease, code in sorted(label_mapping.items(), key=lambda x: x[1]):
        print(f"  {code}: {disease}")
    
    # Also save as JSON for easier debugging
    import json
    with open('data/processed/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save reverse mapping for quick lookup
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    with open('data/processed/reverse_label_mapping.json', 'w') as f:
        json.dump(reverse_mapping, f, indent=2)
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=df['label_encoded']
    )
    
    # Save processed data
    train_df.to_csv(params['data']['processed_train'], index=False)
    test_df.to_csv(params['data']['processed_test'], index=False)
    
    print(f"\nData prepared:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Number of classes: {len(label_mapping)}")
    print(f"  Label distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

if __name__ == "__main__":
    main()