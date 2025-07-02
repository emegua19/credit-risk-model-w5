from fastapi import FastAPI
import mlflow.sklearn
from .pydantic_models import InputData

app = FastAPI()

@app.post("/predict")
async def predict(data: InputData):
    """
    Predicts risk probability for input data.
    """
    # model = mlflow.sklearn.load_model("models:/best_model/1")
    # prediction = model.predict_proba(data.features)[:, 1]
    # return {"risk_probability": prediction.tolist()}
    pass
