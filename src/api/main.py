import logging
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler

# Add root directory to path (2 levels up from src/api)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.api.pydantic_models import CustomerData, PredictionResponse  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Prediction API")

# Load the model (prefer local if available, fallback to MLflow)
MODEL_PATH = "models/GradientBoosting.pkl"  # Relative path from src/api/
RUN_ID = "d7912c083a16407fbe6faf6084698758"  # Update with actual run ID from MLflow
MODEL_NAME = "best_model_GradientBoosting"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from local: {MODEL_PATH}")
else:
    mlflow.set_tracking_uri("file:../mlruns")
    model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/{MODEL_NAME}")
    logger.info(f"Loaded model from MLflow: {MODEL_NAME}")

# Initialize scaler (fit during training, reuse here for consistency)
scaler = StandardScaler()


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerData, customer_id: str = "CustomerId_1"):
    try:
        # Convert input data to DataFrame
        data = [
            [getattr(customer_data, field) for field in customer_data.__fields__.keys()]
        ]
        feature_names = [
            "Recency",
            "Frequency",
            "Monetary",
            "Count_airtime",
            "Count_data_bundles",
            "Count_financial_services",
            "Count_movies",
            "Count_other",
            "Count_ticket",
            "Count_transport",
            "Count_tv",
            "Count_utility_bill",
            "AvgTransactionAmount",
        ]
        df = pd.DataFrame(data, columns=feature_names)

        # Scale the data
        X_scaled = scaler.fit_transform(df)

        # Predict probability of is_high_risk = 1
        probability = model.predict_proba(X_scaled)[0][1]

        return PredictionResponse(
            CustomerId=customer_id,
            is_high_risk_prob=float(probability),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    return {
        "message": (
            "Welcome to the Credit Risk Prediction API. "
            "Use POST /predict for predictions."
        )
    }
