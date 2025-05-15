import torch
from src.ml.models.base_model import BaseModel
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
import numpy as np
from typing import Dict, Any, List
from .base_model import BaseModel
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ALBERTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'albert-base-v2')
        self.max_length = config.get('max_length', 256)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.num_labels = None
        
        # Training parameters
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.epochs = config.get('epochs', 3)
        
        logger.info(f"Initialized ALBERTModel with device: {self.device}")

    def prepare_for_training(self, num_labels: int):
        """Prepare model for training with specific number of labels"""
        self.num_labels = num_labels
        self.model = AlbertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        ).to(self.device)
        logger.info(f"Model prepared for training with {num_labels} labels")

    def preprocess_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare texts for the model"""
        # Handle empty texts
        texts = [text if text else "" for text in texts]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {key: tensor.to(self.device) for key, tensor in encoded.items()}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train the ALBERT model"""
        if self.model is None:
            unique_labels = len(np.unique(y_train))
            self.prepare_for_training(unique_labels)
        
        # Convert to list of strings if needed
        if isinstance(X_train, np.ndarray) and len(X_train.shape) > 0:
            if isinstance(X_train[0], np.ndarray):
                X_train = [str(x) for x in X_train]
            else:
                X_train = X_train.tolist()
        
        # Ensure texts are strings
        X_train = [str(x) for x in X_train]
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        self.model.train()
        metrics = {}
        
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Process in batches
            for i in range(0, len(X_train), self.batch_size):
                batch_texts = X_train[i:i+self.batch_size]
                batch_labels = torch.tensor(y_train[i:i+self.batch_size]).to(self.device)
                
                # Tokenize batch
                inputs = self.preprocess_text(batch_texts)
                
                # Forward pass
                outputs = self.model(**inputs, labels=batch_labels)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
            
            # Calculate epoch metrics
            epoch_loss = total_loss / (len(X_train) // self.batch_size)
            epoch_accuracy = correct / total
            
            metrics[f'epoch_{epoch}_loss'] = epoch_loss
            metrics[f'epoch_{epoch}_accuracy'] = epoch_accuracy
            
            logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
        
        # Final metrics
        metrics['train_accuracy'] = epoch_accuracy
        metrics['train_loss'] = epoch_loss
        
        # Validation if provided
        if X_val is not None and y_val is not None:
            val_accuracy = self.evaluate(X_val, y_val)
            metrics['val_accuracy'] = val_accuracy
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model on data"""
        self.model.eval()
        
        # Convert to list of strings
        if isinstance(X, np.ndarray) and len(X.shape) > 0:
            if isinstance(X[0], np.ndarray):
                X = [str(x) for x in X]
            else:
                X = X.tolist()
        
        X = [str(x) for x in X]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_texts = X[i:i+self.batch_size]
                batch_labels = torch.tensor(y[i:i+self.batch_size]).to(self.device)
                
                inputs = self.preprocess_text(batch_texts)
                outputs = self.model(**inputs)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        return correct / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        self.model.eval()
        
        # Convert to list of strings
        if isinstance(X, np.ndarray) and len(X.shape) > 0:
            if isinstance(X[0], np.ndarray):
                X = [str(x) for x in X]
            else:
                X = X.tolist()
        
        X = [str(x) for x in X]
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_texts = X[i:i+self.batch_size]
                inputs = self.preprocess_text(batch_texts)
                outputs = self.model(**inputs)
                
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        self.model.eval()
        
        # Convert to list of strings
        if isinstance(X, np.ndarray) and len(X.shape) > 0:
            if isinstance(X[0], np.ndarray):
                X = [str(x) for x in X]
            else:
                X = X.tolist()
        
        X = [str(x) for x in X]
        
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_texts = X[i:i+self.batch_size]
                inputs = self.preprocess_text(batch_texts)
                outputs = self.model(**inputs)
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)

    def save_model(self, path: str):
        """Save model and tokenizer"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save custom config
        config_path = os.path.join(path, 'custom_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'num_labels': self.num_labels,
                'max_length': self.max_length,
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs
            }, f)
        
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model and tokenizer"""
        # Load custom config
        config_path = os.path.join(path, 'custom_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            self.num_labels = custom_config.get('num_labels', self.num_labels)
            self.max_length = custom_config.get('max_length', self.max_length)
        
        # Load model and tokenizer
        self.model = AlbertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        
        logger.info(f"Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        info = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': str(self.device),
            'num_labels': self.num_labels,
            'is_loaded': self.model is not None
        }
        
        if self.model is not None:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return info