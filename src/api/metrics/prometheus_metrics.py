from prometheus_client import Counter, Histogram

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Model metrics
MODEL_PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model']
)

MODEL_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency',
    ['model']
)

# Training metrics
TRAINING_RUNS = Counter(
    'model_training_runs_total',
    'Total training runs',
    ['model']
)

MODEL_ACCURACY = Histogram(
    'model_accuracy',
    'Model accuracy scores',
    ['model', 'stage']
)