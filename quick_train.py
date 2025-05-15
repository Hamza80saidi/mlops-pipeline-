import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import yaml
import os

# Load data
df = pd.read_csv('data/raw/symptoms2diseases.csv')

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Create proper label mapping
label_mapping = {label: int(i) for i, label in enumerate(le.classes_)}

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/logistic', exist_ok=True)

# Save label mapping
with open('data/processed/label_mapping.yaml', 'w') as f:
    yaml.dump(label_mapping, f)

print("Label mapping:")
for disease, code in label_mapping.items():
    print(f"  {disease}: {code}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'models/logistic_regression/model.pkl')
joblib.dump(vectorizer, 'models/logistic_regression/vectorizer.pkl')

print("\nModel trained and saved successfully!")
print(f"Training accuracy: {model.score(X_train_vec, y_train):.2f}")
print(f"Test accuracy: {model.score(X_test_vec, y_test):.2f}")

# Test predictions with probabilities
test_text = ["I have a headache and fever"]
test_vec = vectorizer.transform(test_text)
prediction = model.predict(test_vec)[0]
probabilities = model.predict_proba(test_vec)[0]

print(f"\nTest prediction for: '{test_text[0]}'")
print(f"Predicted class: {prediction} -> {le.classes_[prediction]}")
print("\nConfidence scores for all classes:")
for disease, prob in zip(le.classes_, probabilities):
    print(f"  {disease}: {prob:.3f}")

# Save train/test data
train_df = pd.DataFrame({
    'text': X_train,
    'label': [le.classes_[y] for y in y_train],
    'label_encoded': y_train
})
test_df = pd.DataFrame({
    'text': X_test,
    'label': [le.classes_[y] for y in y_test],
    'label_encoded': y_test
})

train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)
