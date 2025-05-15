from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware
from ..core.config import settings
from ..core.database import engine, Base
from .routers import auth, predict, health
from .metrics.prometheus_metrics import REQUEST_COUNT, REQUEST_LATENCY, MODEL_PREDICTION_COUNT
import time

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
        
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
        return response

app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["predictions"])
app.include_router(health.router, prefix=f"{settings.API_V1_STR}/health", tags=["health"])

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
def root():
    return {"message": "Medical Symptom Classification API"}