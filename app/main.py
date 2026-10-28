from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    model_load_error = None
except (EOFError, ImportError, OSError, ValueError) as error:
    model = None
    model_load_error = str(error)

class HeartDiseaseFeatures(BaseModel):
    age: float = Field(
        ..., ge=1, le=120, allow_inf_nan=False, json_schema_extra={"example": 55.0}
    )
    trestbps: float = Field(
        ..., ge=50, le=250, allow_inf_nan=False, json_schema_extra={"example": 130.0}
    )
    chol: float = Field(
        ..., ge=50, le=700, allow_inf_nan=False, json_schema_extra={"example": 240.0}
    )
    thalach: float = Field(
        ..., ge=40, le=250, allow_inf_nan=False, json_schema_extra={"example": 150.0}
    )
    oldpeak: float = Field(
        ..., ge=0, le=10, allow_inf_nan=False, json_schema_extra={"example": 1.5}
    )

class PredictionResponse(BaseModel):
    prediction: int
    result: str

@app.get("/")
def health_check():
    response = {"status": "healthy", "model_ready": model is not None}
    if model_load_error:
        response["model_error"] = model_load_error
    return response

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model file not loaded.")
    
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
