from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import os
import psutil
from datetime import datetime
from ...core.database import get_db

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Medical Symptom Classifier API"
    }

@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Readiness check - verifies all dependencies are available"""
    try:
        # Check database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check if models directory exists
    models_exist = os.path.exists("models")
    
    # Check if at least one model is available
    available_models = []
    if models_exist:
        model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
        for model_dir in model_dirs:
            if os.path.exists(os.path.join("models", model_dir, "model.pkl")):
                available_models.append(model_dir)
    
    ready = db_status == "connected" and len(available_models) > 0
    
    return {
        "ready": ready,
        "database": db_status,
        "models_available": available_models,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check - simple endpoint to verify service is running"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def system_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    # Process info
    process = psutil.Process(os.getpid())
    
    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        },
        "process": {
            "pid": process.pid,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/info")
async def api_info() -> Dict[str, Any]:
    """Get API information"""
    from ...core.config import settings
    
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_version": os.sys.version,
        "api_prefix": settings.API_V1_STR,
        "docs_url": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    }