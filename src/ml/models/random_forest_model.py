from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any
import joblib
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 10),
            min_samples_split=config.get('min_samples_split', 2),
            random_state=config.get('random_state', 42)
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