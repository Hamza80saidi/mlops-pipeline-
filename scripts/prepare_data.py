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
    
    # Load raw data
    df = pd.read_csv(params['data']['raw_data'])
    
    # Basic cleaning
    df = df.dropna()
    df['text'] = df['text'].str.strip()
    
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # Save label mapping
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    with open('data/processed/label_mapping.yaml', 'w') as f:
        yaml.dump(label_mapping, f)
    
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
    
    print(f"Data prepared: {len(train_df)} training samples, {len(test_df)} test samples")

if __name__ == "__main__":
    main()