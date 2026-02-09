"""
FastAPI application.

Exposes health check and prediction endpoints for the ML service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from mlops_api.predict import predict

logger = logging.getLogger(__name__)


app = FastAPI(title="MLOps Mini Prod")

class InputSchema(BaseModel):
    price: float
    promotion: int
    temperature: float
    
    
@app.on_event("startup")
def startup_event():
    logger.info("API startup completed")


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/predict")
def predict_endpoint(payload: InputSchema):
    if payload.price < 0:
        raise HTTPException(
            status_code=400,
            detail="Price must be non-negative"
        )

    return predict(payload.model_dump())
