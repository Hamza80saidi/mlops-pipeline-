from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from ...core.database import get_db
from ..auth.oauth2 import get_current_active_user
from ..auth.models import User
from ...services.prediction_service import PredictionService
from ..metrics.prometheus_metrics import MODEL_PREDICTION_COUNT, MODEL_LATENCY
import time

router = APIRouter()

class PredictionRequest(BaseModel):
    text: str
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str
    all_confidences: Dict[str, float]  # Added this field
    processing_time: Optional[float] = None

class PredictionHistoryResponse(BaseModel):
    id: int
    text: str
    prediction: str
    confidence: float
    model_name: str
    created_at: str
    processing_time: Optional[float] = None

@router.post("/", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make prediction using the specified model"""
    MODEL_PREDICTION_COUNT.labels(model=request.model_name or "default").inc()
    
    start_time = time.time()
    
    prediction_service = PredictionService(db)
    
    try:
        result = prediction_service.predict(
            text=request.text,
            model_name=request.model_name,
            user_id=current_user.id
        )
        
        duration = time.time() - start_time
        MODEL_LATENCY.labels(model=result['model_used']).observe(duration)
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            model_used=result['model_used'],
            all_confidences=result['all_confidences'],
            processing_time=duration
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get user's prediction history"""
    prediction_service = PredictionService(db)
    history = prediction_service.get_user_history(current_user.id, skip, limit)
    
    return [
        PredictionHistoryResponse(
            id=item['id'],
            text=item['text'],
            prediction=item['prediction'],
            confidence=item['confidence'],
            model_name=item['model_name'],
            created_at=item['created_at'],
            processing_time=item.get('processing_time')
        )
        for item in history
    ]

@router.get("/models")
async def get_available_models(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of available models"""
    prediction_service = PredictionService(db)
    return prediction_service.get_available_models()

@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get information about a specific model"""
    prediction_service = PredictionService(db)
    return prediction_service.get_model_info(model_name)