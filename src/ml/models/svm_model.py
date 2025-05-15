from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Any
import joblib
from .base_model import BaseModel

class SVMModel(BaseModel):
    """Support Vector Machine implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.model = SVC(
            C=config.get('C', 1.0),
            kernel=config.get('kernel', 'rbf'),
            max_iter=config.get('max_iter', 1000),
            probability=True
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        metrics = {'train_accuracy': train_score}
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            metrics['val_accuracy'] = val_score
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    def load_model(self, path: str):
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']