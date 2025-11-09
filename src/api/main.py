from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from ..core.config import settings
from ..core.database import engine, Base
from .routers import auth, predict, health
from .metrics.prometheus_metrics import REQUEST_COUNT, REQUEST_LATENCY
import time
import uvicorn

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",

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
        # Don't track metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
            
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

# Create metrics app
metrics_app = make_asgi_app()

# Add metrics endpoint directly to main app
@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/")
def root():
    return {"message": "Medical Symptom Classification API"}

@app.on_event("startup")
async def startup_event():
    pass  # No initialization logic needed for User or PredictionHistory
    from ..repository.prediction_repository import PredictionHistory
    
    Base.metadata.create_all(bind=engine)
    print("Database tables created/verified")
    print(f"Metrics endpoint available at: http://localhost:8000/metrics")

if __name__ == "__main__":
    # Run both servers
    uvicorn.run(app, host="0.0.0.0", port=8000)