# LabProject25
# MLOps Pipeline Flowchart


## 1. Initial Training & Deployment

- **Start:** Manual trigger or first-time run.
- **Data Ingestion:** Raw data is loaded.
- **Preprocessing:** `preprocessing.py` runs a Spark pipeline to clean and transform the data.
- **Model Training:** `train.ipynb or train.py` trains multiple models and logs them to MLflow.
- **Evaluation:** Models are evaluated using cross-validation; metrics (e.g., accuracy) are logged.
- **Model Selection:** `deployment.py` queries MLflow to identify the best-performing model.
- **Model Promotion:** The best model is transitioned to the "Production" stage in MLflow.
- **API Deployment:** `application.py` starts a FastAPI server, loading the production-ready model for real-time inference.


## 2. Continuous Monitoring & Maintenance

- **Prediction Request:** An external application sends a request to the `/predict` endpoint.
- **Inference:** The FastAPI server uses the production model to generate a prediction.
- **Data Buffering:** The incoming request data is added to a buffer for drift detection.
- **Drift Detection:** `application.py` initiates a background task to check for data drift using `drift_detection.py`.
- **Scheduled Retraining:** `scheduler.py` runs a background thread that triggers retraining every 7 days.
- **Drift Detected?**
    - **No:** The system continues to serve predictions.
    - **Yes:**
        - **Trigger Retraining:** A new training job is triggered asynchronously.
        - **New Model Trained:** A new model is trained and evaluated, potentially surpassing the current production model.
        - **Update Production:** The new best model is promoted to the "Production" stage. The FastAPI server automatically loads the new version, ensuring zero downtime.

- **End:** The cycle repeats, ensuring the model is always up-to-date and performant.

