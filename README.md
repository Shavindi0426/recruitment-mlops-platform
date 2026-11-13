# Recruitment Prediction MLOps Platform

### Design and Deploy a Real-Time AI Prediction Platform with an Automated MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) platform that predicts recruitment assessment outcomes using machine learning. The platform integrates model training, experiment tracking, real-time predictions, data drift monitoring, controlled model retraining, Docker containerization, cloud deployment, and continuous integration.

---

## 1. Project Overview

The Recruitment Prediction MLOps Platform demonstrates how a machine learning system can be developed, deployed, monitored, and maintained using modern MLOps practices.

The system predicts whether a recruitment candidate will Pass or Fail based on candidate-related features. It provides a REST API for real-time predictions and a Streamlit dashboard for monitoring model performance and data drift.

The platform also implements a controlled model retraining process that evaluates candidate models against the existing production model before allowing model replacement.

## 2. Project Objectives

- Develop an end-to-end machine learning pipeline.
- Generate and preprocess recruitment candidate data.
- Train and evaluate multiple machine learning models.
- Track experiments and model performance using MLflow.
- Provide real-time predictions through FastAPI.
- Monitor incoming data for distribution changes.
- Visualize monitoring results using Streamlit.
- Implement controlled model retraining with approval conditions.
- Containerize the application using Docker.
- Deploy the prediction API to Railway.
- Automate testing using GitHub Actions.

## 3. System Architecture

The platform consists of the following main components:

1. **Data Ingestion:** Loads recruitment candidate data and incoming feature data.
2. **Data Preprocessing:** Handles missing values, duplicate records, target encoding, data splitting, and feature scaling.
3. **Model Training:** Trains Logistic Regression and Random Forest classifiers.
4. **Model Evaluation:** Evaluates models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
5. **Experiment Tracking:** Uses MLflow to record experiments, parameters, metrics, and model artifacts.
6. **Prediction Service:** Uses FastAPI to provide real-time recruitment predictions.
7. **Drift Monitoring:** Uses PSI and Kolmogorov–Smirnov (KS) statistics to identify changes in incoming feature distributions.
8. **Monitoring Dashboard:** Uses Streamlit to visualize model performance and drift results.
9. **Controlled Retraining:** Evaluates candidate models against the existing production model before replacement.
10. **CI/CD:** Uses GitHub Actions for automated testing.
11. **Containerization:** Uses Docker and Docker Compose to run the API and dashboard in containers.
12. **Cloud Deployment:** Hosts the FastAPI prediction service on Railway.

## 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| API Development | FastAPI, Uvicorn |
| Monitoring Dashboard | Streamlit, Plotly |
| Experiment Tracking | MLflow |
| Drift Detection | PSI, KS Test |
| Model Storage | Joblib |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud Deployment | Railway |
| Testing | Pytest |

## 5. Machine Learning Models

The project compares two classification algorithms:

- Logistic Regression
- Random Forest Classifier

The model with the highest F1 Score is selected as the best model.

### Selected Model Performance

The Logistic Regression model achieved the following results on the project's held-out test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 84.17% |
| Precision | 86.70% |
| Recall | 92.45% |
| F1 Score | 89.48% |
| ROC-AUC | 90.62% |

These results represent the performance on the generated project dataset and should not be interpreted as guaranteed performance on real-world recruitment data.

## 6. Dataset Description

The project uses a generated recruitment dataset containing 3,000 candidate records.

### Input Features

- `experience_years`
- `skills_match_score`
- `assessment_score`
- `industry_experience_years`
- `interview_score`

### Target Variable

- `assessment_outcome`

The target variable represents the candidate's assessment outcome:

- Pass = 1
- Fail = 0

The preprocessing workflow includes duplicate removal, target encoding, stratified train-test splitting, median imputation, and feature scaling.

The separate incoming dataset used for drift monitoring contains feature values only and does not contain target labels.

## 7. MLflow Experiment Tracking

MLflow is integrated to track machine learning experiments.

It records:

- Model parameters
- Evaluation metrics
- Training runs
- Model artifacts
- Model comparisons

The project uses a local SQLite tracking database named `mlflow.db`.

Experiments include:

- `Recruitment_Prediction_Experiments`
- `Recruitment_Automated_Retraining`

## 8. Real-Time Prediction API

FastAPI provides a REST API for recruitment predictions.

**Live API:**  
https://recruitment-api-production-5bd7375e.up.railway.app/

**Interactive Swagger Documentation:**  
https://recruitment-api-production-5bd7375e.up.railway.app/docs

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/api-info` | API information |
| POST | `/predict` | Generate recruitment prediction |

### Example Prediction Request

```json
{
  "experience_years": 5,
  "skills_match_score": 85,
  "assessment_score": 90,
  "industry_experience_years": 4,
  "interview_score": 88
}
```

### Example Prediction Response

```json
{
  "prediction": "Pass",
  "prediction_class": 1,
  "confidence": 99.74,
  "status": "Success"
}
```

The exact prediction and confidence depend on the input and deployed model.

## 9. Data Drift Monitoring

The platform monitors changes in incoming feature distributions by comparing reference data with incoming data.

The monitoring process uses:

- Population Stability Index (PSI)
- Kolmogorov–Smirnov (KS) Test

Monitoring outputs:

- `monitoring/drift_report.json`
- `monitoring/drift_history.csv`

The dashboard displays feature-level drift results, statistics, visualizations, and drift alerts.

The demonstrated incoming dataset was deliberately shifted to test drift detection. It is a monitoring test sample, not a live production data stream.

## 10. Monitoring Dashboard

The Streamlit dashboard provides an interactive interface for monitoring the platform.

### Dashboard Sections

1. Dashboard Overview
2. Data Drift Monitoring
3. Feature Analysis
4. Model Performance
5. Monitoring History

The dashboard displays:

- Overall drift status
- Total and drifted features
- PSI and KS statistics
- Feature distribution comparisons
- Model evaluation metrics
- Drift alerts
- Monitoring history

### Run the Dashboard Locally

```bash
streamlit run dashboard/app.py
```

Open:

http://localhost:8501

## 11. Controlled Automated Retraining

The retraining process is implemented in:

`src/retrain.py`

It trains candidate models using labelled recruitment data and compares the best candidate against the existing production model.

### Model Approval Conditions

A candidate model is approved only when all the following conditions are satisfied:

1. F1 Score improves by at least 0.01.
2. Accuracy does not decrease.
3. ROC-AUC does not decrease.

If the candidate fails any condition, the existing production model is retained.

### Run Retraining

```bash
python src/retrain.py
```

**Important:** The current retraining process is manually triggered. A scheduled or event-driven retraining trigger has not yet been configured and validated. Unlabelled drift-monitoring data is not used as supervised training data.

## 12. Dockerization

The application is containerized using Docker and Docker Compose.

The project contains:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `requirements-docker.txt`

Docker Compose runs two services:

- Recruitment FastAPI prediction service
- Streamlit monitoring dashboard

### Build and Start Containers

```bash
docker compose build
docker compose up -d
```

### Check Running Containers

```bash
docker compose ps
```

### View Container Logs

```bash
docker compose logs
```

### Stop Containers

```bash
docker compose down
```

### Local Docker URLs

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

The API and dashboard containers have been successfully built and tested locally.

## 13. CI/CD with GitHub Actions

GitHub Actions is integrated to automate project testing when code changes are pushed to the repository.

The CI workflow is located at:

`.github/workflows/`

### Automated Testing

The workflow performs the following steps:

1. Checks out the repository.
2. Sets up Python.
3. Upgrades pip.
4. Installs project dependencies.
5. Installs testing dependencies.
6. Runs automated tests using Pytest.

The latest verified workflow completed successfully with:

**11 automated tests passed.**

The tests cover API endpoints, prediction functionality, model files, preprocessing artifacts, and model loading.

View the workflow results in the GitHub Actions tab.

## 14. Project Structure

```text
recruitment-mlops-platform/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── incoming/
│
├── models/
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── backups/
│
├── monitoring/
│   ├── drift_report.json
│   └── drift_history.csv
│
├── reports/
│   ├── model_comparison.csv
│   ├── retraining_comparison.csv
│   └── retraining_decision.txt
│
├── src/
│   ├── preprocessing.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── drift_detection.py
│   ├── generate_drift_data.py
│   └── retrain.py
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── main.py
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── README.md
└── mlflow.db
```

Some generated data, model artifacts, and local tracking files may be excluded from Git depending on repository configuration.

## 15. Installation and Local Execution

### Prerequisites

- Python 3.13
- Git
- Docker Desktop (for containerized execution)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Shavindi0426/recruitment-mlops-platform.git
```

### Step 2: Open the Project Directory

```bash
cd recruitment-mlops-platform
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the End-to-End Pipeline

```bash
python main.py
```

### Step 7: Start the FastAPI Service

```bash
uvicorn api.main:app --reload
```

Open:

http://127.0.0.1:8000/docs

### Step 8: Start the Streamlit Dashboard

Open a separate terminal and run:

```bash
streamlit run dashboard/app.py
```

Open:

http://localhost:8501

### Step 9: Run Automated Tests

```bash
pytest -v tests/
```

### Step 10: Run Controlled Retraining

```bash
python src/retrain.py
```

Run retraining only when genuine labelled data is available.

## 16. Project Verification Status

| Component | Verification |
|---|---|
| End-to-end ML pipeline | Successfully executed locally |
| Model training and evaluation | Verified |
| MLflow experiment tracking | Verified locally |
| FastAPI prediction service | Successfully deployed and tested |
| Data drift detection | Verified locally |
| Streamlit monitoring dashboard | Working locally |
| Controlled retraining | Verified locally |
| Docker API and dashboard | Successfully built and run locally |
| GitHub Actions CI | 11 tests passed |
| Railway API deployment | Active and verified |
| Scheduled retraining | Not yet configured |

## 17. Future Improvements

- Configure scheduled or event-driven retraining using genuine labelled data.
- Deploy the Streamlit dashboard as a separate public cloud service.
- Integrate a continuous incoming data source.
- Introduce production-grade monitoring and alert notifications.
- Improve model validation using newly collected labelled recruitment data.
- Implement secure authentication and role-based API access.
- Introduce model versioning and a centralized model registry.

## 18. Conclusion

The Recruitment Prediction MLOps Platform demonstrates an end-to-end machine learning workflow integrating data preprocessing, model development, experiment tracking, real-time prediction, data drift monitoring, controlled retraining, containerization, cloud deployment, and automated testing.

The project provides a foundation for developing and maintaining machine learning prediction services using modern MLOps practices.

---

## Author

**Individual Academic Project**

Project Duration: 29 October – 12 November 2026

GitHub Repository:  
https://github.com/Shavindi0426/recruitment-mlops-platform

Live API:  
https://recruitment-api-production-5bd7375e.up.railway.app/

API Documentation:  
https://recruitment-api-production-5bd7375e.up.railway.app/docs
