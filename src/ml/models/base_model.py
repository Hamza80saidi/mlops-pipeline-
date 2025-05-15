from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any
import mlflow
from sklearn.base import BaseEstimator

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """Load model from disk"""
        pass
    
    def log_to_mlflow(self, metrics: Dict[str, float], params: Dict[str, Any]):
        """Log model and metrics to MLflow"""
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            if isinstance(self.model, BaseEstimator):
                mlflow.sklearn.log_model(self.model, self.model_name)
            else:
                # For deep learning models
                mlflow.pytorch.log_model(self.model, self.model_name)