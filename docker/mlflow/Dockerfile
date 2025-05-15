FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install mlflow boto3 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]