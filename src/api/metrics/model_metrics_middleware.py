from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge
import time
import json

# Model-specific counters
MODEL_REQUESTS = Counter(
    'model_api_requests_total',
    'Total requests to model endpoints',
    ['endpoint', 'method', 'model']
)

MODEL_REQUEST_DURATION = Histogram(
    'model_api_request_duration_seconds',
    'Request duration for model endpoints',
    ['endpoint', 'model']
)

MODEL_ERRORS = Counter(
    'model_api_errors_total',
    'Total errors in model endpoints',
    ['endpoint', 'model', 'error_type']
)

# Active predictions gauge
ACTIVE_PREDICTIONS = Gauge(
    'model_active_predictions',
    'Number of predictions currently being processed',
    ['model']
)

async def track_model_metrics(request: Request, call_next):
    """Middleware to track model-specific metrics"""
    
    # Only track prediction endpoints
    if "/predict" not in request.url.path:
        return await call_next(request)
    
    # Try to extract model name from request
    model_name = "unknown"
    if request.method == "POST":
        try:
            # Read body to get model name
            body = await request.body()
            request._body = body  # Store for later use
            data = json.loads(body)
            model_name = data.get("model_name", "default")
        except:
            pass
    
    # Track active predictions
    ACTIVE_PREDICTIONS.labels(model=model_name).inc()
    
    # Time the request
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Track successful request
        MODEL_REQUESTS.labels(
            endpoint=request.url.path,
            method=request.method,
            model=model_name
        ).inc()
        
        # Track duration
        duration = time.time() - start_time
        MODEL_REQUEST_DURATION.labels(
            endpoint=request.url.path,
            model=model_name
        ).observe(duration)
        
        return response
        
    except Exception as e:
        # Track errors
        MODEL_ERRORS.labels(
            endpoint=request.url.path,
            model=model_name,
            error_type=type(e).__name__
        ).inc()
        raise
        
    finally:
        # Decrement active predictions
        ACTIVE_PREDICTIONS.labels(model=model_name).dec()