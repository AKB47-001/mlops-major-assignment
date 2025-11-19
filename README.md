# MLOps Major Project  
Face Classification using Olivetti Dataset - MLOps Major Assignment

## 1. Project Description
This project trains a Decision Tree model on the Olivetti Faces dataset, tests the model, containerizes the app using Docker, and deploys it on Kubernetes with 3 replicas. A simple Flask web app is used for image upload and prediction.

## 2. Branch Structure
- **main** – initial setup only  
- **dev** – training, testing scripts + GitHub Actions CI  
- **docker_cicd** – Flask app, Dockerfile, Kubernetes deployment  

## 3. How to Run Locally

### Install dependencies:
```bash
pip install -r requirements.txt
