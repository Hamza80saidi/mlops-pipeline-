from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class TrainingStrategy(ABC):
    """Abstract strategy for training models"""
    
    @abstractmethod
    def execute(self, model: 'BaseModel', X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None) -> Tuple['BaseModel', Dict[str, float]]:
        """Execute the training strategy"""
        pass

class StandardTrainingStrategy(TrainingStrategy):
    """Standard training strategy for traditional ML models"""
    
    def execute(self, model: 'BaseModel', X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None) -> Tuple['BaseModel', Dict[str, float]]:
        metrics = model.train(X_train, y_train, X_val, y_val)
        return model, metrics

class TransferLearningStrategy(TrainingStrategy):
    """Strategy for fine-tuning pre-trained models"""
    
    def execute(self, model: 'BaseModel', X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None) -> Tuple['BaseModel', Dict[str, float]]:
        # Specific implementation for transfer learning
        if hasattr(model, 'prepare_for_training'):
            model.prepare_for_training()
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        return model, metrics