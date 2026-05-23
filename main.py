from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("isolation_forest_model.pkl")
category_encoder = joblib.load("category_encoder.pkl")


class Transaction(BaseModel):
    amount: int
    category: str


class TransactionRequest(BaseModel):
    transactions: List[Transaction]


@app.post("/detect-anomaly")
def detect_anomaly(request: TransactionRequest):

    txn = request.transactions[0]

    category_encoded = category_encoder.transform(
        [txn.category]
    )[0]

    df = pd.DataFrame([{
        "amount": txn.amount,
        "category_encoded": category_encoded
    }])

    prediction = model.predict(df)[0]

    score = model.decision_function(df)[0]

    return {
        "anomaly": True if prediction == -1 else False,
        "score": round(float(abs(score)), 2)
    }