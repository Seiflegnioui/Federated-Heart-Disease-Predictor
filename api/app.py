from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import torch
import os
import sys
import numpy as np
import pandas as pd

# Add src to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from model import HeartDiseaseMLP

app = FastAPI(title="Federated Heart Disease Prediction API")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "global_model_fedprox_noniid.pth")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "processed", "scaler.pkl")

# Load model and scaler
model = HeartDiseaseMLP(input_dim=23)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

scaler = joblib.load(SCALER_PATH)

# Define request body
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

def preprocess(data: PatientData):
    # Match the preprocessing steps from data_preparation.py
    df = pd.DataFrame([data.dict()])
    
    # Categorical columns
    CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
    CONTINUOUS_COLS  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    
    # We need to enforce one-hot encoding exactly as the training set (23 features)
    # The expected categories for each column:
    categories = {
        'cp': [0, 1, 2, 3],
        'restecg': [0, 1, 2],
        'slope': [0, 1, 2],
        'thal': [0, 1, 2, 3]
    }
    
    # Initialize encoded dictionary
    encoded = {
        'age': data.age,
        'sex': data.sex,
        'trestbps': data.trestbps,
        'chol': data.chol,
        'fbs': data.fbs,
        'thalach': data.thalach,
        'exang': data.exang,
        'oldpeak': data.oldpeak,
        'ca': data.ca
    }
    
    # One-hot encode
    for col, cats in categories.items():
        val = getattr(data, col)
        for cat in cats:
            encoded[f"{col}_{cat}"] = 1 if val == cat else 0
            
    # Create final DataFrame in the exact order of feature_names.txt
    feature_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca', 
                     'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0', 'restecg_1', 'restecg_2', 
                     'slope_0', 'slope_1', 'slope_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3']
    
    final_df = pd.DataFrame([encoded], columns=feature_names)
    
    # Scale continuous
    final_df[CONTINUOUS_COLS] = scaler.transform(final_df[CONTINUOUS_COLS])
    
    return torch.tensor(final_df.values.astype(np.float32))

@app.post("/predict")
def predict(data: PatientData):
    inputs = preprocess(data)
    with torch.no_grad():
        outputs = model(inputs)
        prob = torch.sigmoid(outputs).item()
        prediction = 1 if prob >= 0.5 else 0
        
    return {
        "prediction": prediction,
        "probability": prob,
        "status": "Sick" if prediction == 1 else "Healthy"
    }

# Serve Static files for UI
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
