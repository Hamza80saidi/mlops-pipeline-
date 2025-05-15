from typing import Dict, Any
from ..models.base_model import BaseModel
from ..models.svm_model import SVMModel
from ..models.random_forest_model import RandomForestModel
from ..models.logistic_regression_model import LogisticRegressionModel
from ..models.albert_model import ALBERTModel

class ModelFactory:
    """Factory class for creating model instances"""
    
    _models = {
        'svm': SVMModel,
        'random_forest': RandomForestModel,
        'logistic_regression': LogisticRegressionModel,
        'albert': ALBERTModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """Create a model instance based on type"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(config)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register a new model type"""
        cls._models[model_type] = model_class