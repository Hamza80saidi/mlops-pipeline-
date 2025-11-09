📊 MLOps Pipeline for Text Classification

## 📌 Project Overview
This project implements a **complete MLOps pipeline** for text classification.  
It demonstrates how to industrialize the lifecycle of NLP models with a focus on **deployment, security, monitoring, and scalability**.

---

## 🎯 Objectives
- Build a modular and reusable MLOps pipeline.  
- Manage dataset and model versions using **DVC**.  
- Track experiments and metrics with **MLflow**.  
- Containerize and orchestrate services with **Docker** and **Kubernetes**.  
- Monitor system health and performance with **Prometheus** and **Grafana**.  
- Ensure security with **OAuth2/JWT** and **RBAC**.  

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Frameworks**: Hugging Face Transformers, Scikit-learn, FastAPI  
- **Versioning**: Git, GitHub, DVC  
- **Deployment**: Docker, Kubernetes  
- **Monitoring**: Prometheus, Grafana  
- **CI/CD**: GitHub Actions  

---

## 📂 Project Structure
mlops-pipeline/ 
│── configs/             # Configuration files 
│── data/                # Datasets (managed with DVC) 
│── docker/              # Dockerfiles and build scripts 
│── experiments/         # MLflow experiment tracking 
│── kubernetes/          # Kubernetes manifests 
│── models/              # Trained models 
│── reports/             # Reports and metrics 
│── scripts/             # Utility scripts 
│── src/                 # Source code (preprocessing,training, API) 
│── tests/               # Unit tests 
│── docker-compose.yaml 
│── dvc.yaml
│── params.yaml 
│── requirements.txt 
│── run_services.sh 


---

## ⚙️ Installation & Usage

### 1. Clone the repository

git clone https://github.com/yourusername/mlops-pipeline-.git
cd mlops-pipeline
### 2. Create a virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
### 3. Install dependencies

pip install -r requirements.txt
### 4. Run a quick training

python quick_train.py
###  5. Start services (API + Monitoring)

./run_services.sh

###  📊 Results
Logistic Regression: 95.8% accuracy on validation

BERT: >90% accuracy on complex texts

SVM: strong performance on imbalanced datasets

API response time: 116 ms average

Monitoring dashboards available via Grafana and Prometheus

### 🔒 Security
Authentication with OAuth2 + JWT

Role-based access control (RBAC)

Encrypted communication

###  🚀 Future Improvements
Cloud deployment (AWS/GCP/Azure)
Multilingual support (XLM-RoBERTa)
Continuous retraining with drift detection
Automated CI/CD pipeline
A/B testing for new models

###  Author
Developed by HAMZA SAIDI - AKRAM TAYEB - EL MEHDI EL MAHLALI Contact: saidih794@gmail.com

