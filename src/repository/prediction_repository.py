from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, desc
from ..api.auth.models import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    model_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)  # Time taken for prediction
    
    class Config:
        from_attributes = True

class PredictionRepository:
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def save_prediction(self, user_id: int, text: str, prediction: str,
                        confidence: float, model_name: str, 
                        processing_time: float = None) -> PredictionHistory:
        """Save prediction to history"""
        prediction_record = PredictionHistory(
            user_id=user_id,
            text=text,
            prediction=prediction,
            confidence=confidence,
            model_name=model_name,
            processing_time=processing_time
        )
        
        self.db_session.add(prediction_record)
        self.db_session.commit()
        self.db_session.refresh(prediction_record)
        
        return prediction_record
    
    def get_user_predictions(self, user_id: int, skip: int = 0, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get user's prediction history"""
        predictions = self.db_session.query(PredictionHistory)\
            .filter(PredictionHistory.user_id == user_id)\
            .order_by(desc(PredictionHistory.created_at))\
            .offset(skip)\
            .limit(limit)\
            .all()
        
        return [
            {
                "id": p.id,
                "text": p.text,
                "prediction": p.prediction,
                "confidence": p.confidence,
                "model_name": p.model_name,
                "created_at": p.created_at.isoformat(),
                "processing_time": p.processing_time
            }
            for p in predictions
        ]
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[PredictionHistory]:
        """Get specific prediction by ID"""
        return self.db_session.query(PredictionHistory)\
            .filter(PredictionHistory.id == prediction_id)\
            .first()
    
    def get_predictions_by_model(self, model_name: str, 
                                 limit: int = 100) -> List[PredictionHistory]:
        """Get predictions made by a specific model"""
        return self.db_session.query(PredictionHistory)\
            .filter(PredictionHistory.model_name == model_name)\
            .order_by(desc(PredictionHistory.created_at))\
            .limit(limit)\
            .all()
    
    def get_prediction_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get prediction statistics"""
        query = self.db_session.query(PredictionHistory)
        
        if user_id:
            query = query.filter(PredictionHistory.user_id == user_id)
        
        total_predictions = query.count()
        
        # Get model usage stats
        from sqlalchemy import func
        model_stats = query.group_by(PredictionHistory.model_name)\
            .with_entities(
                PredictionHistory.model_name,
                func.count(PredictionHistory.id).label('count')
            ).all()
        
        # Get average confidence by model
        confidence_stats = query.group_by(PredictionHistory.model_name)\
            .with_entities(
                PredictionHistory.model_name,
                func.avg(PredictionHistory.confidence).label('avg_confidence')
            ).all()
        
        return {
            "total_predictions": total_predictions,
            "model_usage": {stat[0]: stat[1] for stat in model_stats},
            "average_confidence": {stat[0]: float(stat[1]) for stat in confidence_stats}
        }