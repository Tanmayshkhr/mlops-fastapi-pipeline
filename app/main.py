from pathlib import Path
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

class HeartDiseaseFeatures(BaseModel):
    age: float = Field(..., example=55.0)
    trestbps: float = Field(..., example=130.0)
    chol: float = Field(..., example=240.0)
    thalach: float = Field(..., example=150.0)
    oldpeak: float = Field(..., example=1.5)

class PredictionResponse(BaseModel):
    prediction: int
    result: str

@app.get("/")
def health_check():
    return {"status": "healthy", "model_ready": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model file not loaded.")
    
    # Input formatted with exact feature names expected by the model
    input_df = pd.DataFrame([{
        "age": payload.age,
        "trestbps": payload.trestbps,
        "chol": payload.chol,
        "thalach": payload.thalach,
        "oldpeak": payload.oldpeak
    }])
    
    pred = int(model.predict(input_df)[0])
    result_text = "Heart Disease Detected" if pred == 1 else "Normal / No Disease"
    return PredictionResponse(prediction=pred, result=result_text)
