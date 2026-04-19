from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")


class InputData(BaseModel):
    data: list
    
    
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

# Get metrics
@app.get("/metrics")
def get_metrics():
    with open("metrics.json") as f:
        return json.load(f)

# Predict loan
@app.post("/predict")
def predict(input: InputData):
    data = np.array(input.data).reshape(1, -1)
    prediction = model.predict(data)

    return {
        "prediction": int(prediction[0]),
        "result": "Approved" if prediction[0] == 1 else "Rejected"
    }