import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="Fraud Detection API", version="1.0")

# Enable CORS so your Streamlit app can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend domain (like Streamlit Cloud)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load("model.joblib")

# Explicit list of all columns expected by the model pipeline
ALL_FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

class TransactionInput(BaseModel):
    features: Dict[str, float]

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict_fraud(data: TransactionInput):
    try:
        df = pd.DataFrame([data.features])
        
        # Populate any missing columns with 0.0 and enforce column ordering
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df[ALL_FEATURES]
            
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        
        return {
            "is_fraud": prediction == 1,
            "fraud_probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
