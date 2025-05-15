from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Dict, Any
import joblib
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = LogisticRegression(
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 1000),
            solver=config.get('solver', 'lbfgs')
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        metrics = {'train_accuracy': train_score}
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def save_model(self, path: str):
        joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        self.model = joblib.load(path)