# Create a custom network for better communication
docker network inspect mlops-network >/dev/null 2>&1 || docker network create mlops-network

# MinIO - S3 storage
docker run -d \
  --name mlops-minio \
  --network mlops-network \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -v minio_data:/data \
  minio/minio:latest server /data --console-address ":9001"

# Wait for MinIO to start and create buckets
sleep 5
docker exec mlops-minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec mlops-minio mc mb --ignore-existing local/mlflow
docker exec mlops-minio mc mb --ignore-existing local/dvc

# MLflow - Model tracking (connects to MinIO)
docker run -d \
  --name mlops-mlflow \
  --network mlops-network \
  -p 5000:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  -e AWS_ACCESS_KEY_ID=admin \
  -e AWS_SECRET_ACCESS_KEY=admin \
  -v mlflow_data:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root s3://mlflow \
  --host 0.0.0.0

docker run -d \
  --name mlops-prometheus \
  --network mlops-network \
  -p 9090:9090 \
  -v "$(pwd)/configs/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" \
  -v prometheus_data:/prometheus \
  prom/prometheus:latest

# Grafana - Visualization (connects to Prometheus)

docker run -d \
  --name mlops-grafana \
  --network mlops-network \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  -v $(pwd -W)/configs/grafana/datasources.yaml:/etc/grafana/datasources.yaml \
  -v grafana_data:/var/lib/grafana \
  grafana/grafana:latest