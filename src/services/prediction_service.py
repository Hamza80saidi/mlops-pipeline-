from typing import Dict, Any, List, Optional
import mlflow
from sqlalchemy.orm import Session
from ..ml.strategies.model_factory import ModelFactory
from ..ml.preprocessing.text_preprocessor import TextPreprocessor
from ..core.config import settings
from ..repository.prediction_repository import PredictionRepository
import numpy as np
import joblib
import os
import logging
import yaml

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.prediction_repo = PredictionRepository(db_session)
        self.text_preprocessor = TextPreprocessor()
        self._models_cache = {}
        self._vectorizers_cache = {}
        self._label_mapping_cache = None
    
    def predict(self, text: str, model_name: str = None, user_id: int = None) -> Dict[str, Any]:
        """Make prediction using specified model and return confidence for all classes"""
        
        # Use default model if not specified
        if not model_name:
            model_name = settings.DEFAULT_MODEL_NAME
        
        try:
            # Get model from cache or load it
            model = self._get_or_load_model(model_name)
            
            # Preprocess text based on model type
            if model_name != 'albert':
                # For traditional ML models
                processed_text = self.text_preprocessor.preprocess(text)
                # Convert to TF-IDF features
                features = self._text_to_features(processed_text, model_name)
            else:
                # ALBERT handles its own preprocessing
                features = np.array([text])
            
            # Make prediction
            prediction = model.predict(features)
            
            # Get confidence scores for all classes
            confidence_scores = self._get_all_confidences(model, features)
            
            # Map prediction to label and get all class names
            predicted_label = self._get_label_from_prediction(prediction[0])
            all_labels = self._get_all_labels()
            
            # Create confidence dictionary with disease names
            class_confidences = {}
            for class_idx, confidence in enumerate(confidence_scores[0]):
                disease_name = self._get_label_from_prediction(class_idx)
                class_confidences[disease_name] = float(confidence)
            
            # Sort by confidence
            sorted_confidences = dict(sorted(class_confidences.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
            
            # Get top confidence
            top_confidence = confidence_scores[0][prediction[0]]
            
            # Save prediction history
            if user_id:
                self.prediction_repo.save_prediction(
                    user_id=user_id,
                    text=text,
                    prediction=predicted_label,
                    confidence=float(top_confidence),
                    model_name=model_name
                )
            
            return {
                "prediction": predicted_label,
                "confidence": float(top_confidence),
                "model_used": model_name,
                "all_confidences": sorted_confidences
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def _get_all_confidences(self, model, features: np.ndarray) -> np.ndarray:
        """Get confidence scores for all classes"""
        try:
            if hasattr(model.model, 'predict_proba'):
                # Sklearn models
                return model.model.predict_proba(features)
            elif hasattr(model, 'predict_proba'):
                # Custom models that implement predict_proba
                return model.predict_proba(features)
            else:
                # For models without probability, create one-hot encoded confidence
                predictions = model.predict(features)
                num_classes = len(self._get_all_labels())
                confidences = np.zeros((len(predictions), num_classes))
                for i, pred in enumerate(predictions):
                    confidences[i, pred] = 1.0
                return confidences
        except Exception as e:
            logger.warning(f"Could not get confidence scores: {e}")
            # Return equal confidence for all classes
            num_classes = len(self._get_all_labels())
            return np.ones((1, num_classes)) / num_classes
    
    def _get_label_mapping(self) -> Dict[str, int]:
        """Load and cache label mapping"""
        if self._label_mapping_cache is None:
            try:
                with open('data/processed/label_mapping.yaml', 'r') as f:
                    self._label_mapping_cache = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading label mapping: {e}")
                self._label_mapping_cache = {}
        return self._label_mapping_cache
    
    def _get_label_from_prediction(self, prediction: int) -> str:
        """Map numeric prediction to label name"""
        label_mapping = self._get_label_mapping()
        
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        # Return actual disease name
        return reverse_mapping.get(prediction, f"Unknown_Disease_{prediction}")
    
    def _get_all_labels(self) -> List[str]:
        """Get all available disease labels"""
        label_mapping = self._get_label_mapping()
        return list(label_mapping.keys())
    
    def _get_or_load_model(self, model_name: str):
        """Get model from cache or load from MLflow/disk"""
        if model_name not in self._models_cache:
            try:
                # First try to load from MLflow registry
                model = self._load_from_mlflow(model_name)
            except Exception as e:
                logger.warning(f"Failed to load from MLflow: {e}")
                # Fallback to loading from disk
                model = self._load_from_disk(model_name)
            
            self._models_cache[model_name] = model
        
        return self._models_cache[model_name]
    
    def _load_from_disk(self, model_name: str):
        """Load model from disk"""
        model_path = f"models/{model_name}/model.pkl"
        
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        # Create model instance
        model_config = self._get_model_config(model_name)
        model = ModelFactory.create_model(model_name, model_config)
        
        # Load the model
        model.load_model(model_path)
        
        return model
    
    def _text_to_features(self, text: str, model_name: str) -> np.ndarray:
        """Convert text to features using the appropriate vectorizer"""
        # Get or load vectorizer
        vectorizer = self._get_or_load_vectorizer(model_name)
        
        # Transform text
        features = vectorizer.transform([text])
        
        # Convert to array if needed
        if hasattr(features, 'toarray'):
            features = features.toarray()
        
        return features
    
    def _get_or_load_vectorizer(self, model_name: str):
        """Get or load the vectorizer for a model"""
        if model_name not in self._vectorizers_cache:
            vectorizer_path = f"models/{model_name}/vectorizer.pkl"
            
            if os.path.exists(vectorizer_path):
                # Load saved vectorizer
                self._vectorizers_cache[model_name] = joblib.load(vectorizer_path)
            else:
                # Create a new vectorizer (this should match training)
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=5000)
                
                # Load training data to fit vectorizer
                import pandas as pd
                train_df = pd.read_csv('data/processed/train.csv')
                processed_texts = [self.text_preprocessor.preprocess(text) 
                                   for text in train_df['text'].values]
                vectorizer.fit(processed_texts)
                
                # Save for future use
                os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
                joblib.dump(vectorizer, vectorizer_path)
                
                self._vectorizers_cache[model_name] = vectorizer
        
        return self._vectorizers_cache[model_name]
    
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration from params.yaml"""
        try:
            import yaml
            with open('params.yaml', 'r') as f:
                params = yaml.safe_load(f)
            
            return params.get('models', {}).get(model_name, {})
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
            return {}
    
    def get_user_history(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get user's prediction history"""
        return self.prediction_repo.get_user_predictions(user_id, skip, limit)
    
    def _load_from_mlflow(self, model_name: str):
        """Load model from MLflow registry"""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        client = mlflow.tracking.MlflowClient()
        
        # Get latest model version
        try:
            model_versions = client.get_latest_versions(
                name=f"{settings.mlflow_registry_name}-{model_name}",
                stages=[settings.MODEL_STAGE]
            )
            
            if not model_versions:
                raise Exception(f"No {settings.MODEL_STAGE} version found for model {model_name}")
            
            model_version = model_versions[0]
            model_uri = f"models:/{settings.mlflow_registry_name}-{model_name}/{model_version.version}"
            
        except Exception:
            # Try without stage filter
            model_versions = client.search_model_versions(
                filter_string=f"name='{settings.mlflow_registry_name}-{model_name}'"
            )
            
            if not model_versions:
                raise Exception(f"No versions found for model {model_name}")
            
            # Get latest version
            model_version = sorted(model_versions, key=lambda x: x.version, reverse=True)[0]
            model_uri = f"models:/{settings.mlflow_registry_name}-{model_name}/{model_version.version}"
        
        # Load the model based on type
        if model_name in ['svm', 'random_forest', 'logistic_regression']:
            loaded_model = mlflow.sklearn.load_model(model_uri)
        else:  # ALBERT
            loaded_model = mlflow.pytorch.load_model(model_uri)
        
        # Create model instance
        model_config = self._get_model_config(model_name)
        model = ModelFactory.create_model(model_name, model_config)
        model.model = loaded_model
        
        return model
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about a model"""
        if not model_name:
            model_name = settings.DEFAULT_MODEL_NAME
        
        try:
            model = self._get_or_load_model(model_name)
            
            info = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "is_loaded": model.model is not None
            }
            
            # Add model-specific info
            if hasattr(model, 'get_info'):
                info.update(model.get_info())
            
            return info
            
        except Exception as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "is_loaded": False
            }
    
    def reload_model(self, model_name: str) -> bool:
        """Force reload a model from storage"""
        try:
            # Remove from cache
            if model_name in self._models_cache:
                del self._models_cache[model_name]
            
            # Try to load again
            self._get_or_load_model(model_name)
            
            return True
        except Exception as e:
            logger.error(f"Failed to reload model {model_name}: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        
        # Check MLflow registry
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            
            registered_models = client.list_registered_models()
            
            for rm in registered_models:
                if rm.name.startswith(settings.mlflow_registry_name):
                    model_name = rm.name.replace(f"{settings.mlflow_registry_name}-", "")
                    latest_version = rm.latest_versions[0] if rm.latest_versions else None
                    
                    models.append({
                        "name": model_name,
                        "source": "mlflow",
                        "version": latest_version.version if latest_version else None,
                        "stage": latest_version.current_stage if latest_version else None
                    })
        except Exception as e:
            logger.warning(f"Could not list MLflow models: {e}")
        
        # Check local disk
        models_dir = "models"
        if os.path.exists(models_dir):
            for model_name in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_name, "model.pkl")
                if os.path.exists(model_path):
                    # Check if already in list
                    if not any(m['name'] == model_name for m in models):
                        models.append({
                            "name": model_name,
                            "source": "disk",
                            "path": model_path
                        })
        
        return models