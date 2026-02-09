"""
FastAPI application.

Exposes health check and prediction endpoints for the ML service.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from mlops_api.predict import predict


app = FastAPI(title="MLOps Mini Prod")


class Features(BaseModel):
    price: float
    promotion: int
    temperature: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(f: Features):
    return predict(f.model_dump())


