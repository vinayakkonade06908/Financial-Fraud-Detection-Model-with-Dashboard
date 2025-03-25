import logging
import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.preprocessing import preprocess_data
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\transactions.csv")
df, _ = preprocess_data(df)
print(df.head)

# Load Model & Scaler Paths
MODEL_PATH = "C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\backend\\models\\fraud_model.pkl"
SCALER_PATH = "C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\backend\\models\\scaler.pkl"

# Ensure Model & Scaler Exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file '{MODEL_PATH}' not found!")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"🚨 Scaler file '{SCALER_PATH}' not found!")

# Load Model & Scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Model & Scaler Loaded Successfully!")

df_processed = preprocess_data(df, scaler)

# Define API App
app = FastAPI(debug=True)

print(f"Received JSON:")

# Class to Validate Incoming Transactions
class Transaction(BaseModel):
    amount: float
    transaction_type: str
    merchant_category: str
    transaction_count: int

@app.post("/predict/")
def predict_fraud(transaction: Transaction):
    """ Predict if a transaction is fraudulent or safe. """
    try:
        # Convert transaction to DataFrame
        data = pd.DataFrame([transaction.dict()])
        print("\n🔹 Raw Input Data:\n", data)

        # Preprocess Input Data
        data, _ = preprocess_data(data, scaler)
        print("\n🔹 Processed Data Before Prediction:\n", data)

        # Ensure column order matches trained model
        expected_features = model.feature_names_in_
        data = data[expected_features]  # Reorder columns

        # Predict Fraud Probability
        fraud_prob = model.predict_proba(data)[:, 1]
        print("\n🔹 Fraud Probability:", fraud_prob)

        # Set Threshold (Adjustable)
        threshold = 0.4  # Adjust based on model performance
        is_fraud = (fraud_prob > threshold).astype(int)

        return {
            "fraud": bool(is_fraud[0]),
            "fraud_probability": float(fraud_prob[0])
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to Fraud Detection API"}

# Load transactions.csv for testing
CSV_FILE = "C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\transactions.csv"

@app.get("/transactions/")
async def get_transactions():
    """ Fetch transactions from the CSV file. """
    try:
        df = pd.read_csv(CSV_FILE)  # Read CSV
        if df.empty:
            return JSONResponse(content={"error": "CSV file is empty"}, status_code=404)

        return df.to_dict(orient="records")  # Convert to JSON List

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Logging Configuration
logging.basicConfig(level=logging.DEBUG)