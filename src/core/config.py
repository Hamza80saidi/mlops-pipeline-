from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    mlflow_registry_name: str = "medical-symptom-classifier"

    PROJECT_NAME: str = "Medical Symptom Classifier"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = "sqlite:///./users.db"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"
    MLFLOW_BUCKET: str = "mlflow"
    
    # MinIO
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_ENDPOINT: str = "http://localhost:9000"

    # DVC
    DVC_REMOTE_URL: str = "s3://dvc/data"
    
    # Model Configuration
    DEFAULT_MODEL_NAME: str = "random_forest"
    MODEL_STAGE: str = "Production"
    
    # Prometheus
    PROMETHEUS_PORT: int = 9090

    class Config:
        env_file = ".env"

settings = Settings()