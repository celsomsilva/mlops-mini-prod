"""
FastAPI application.

Exposes health check and prediction endpoints for the ML service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from mlops_api.predict import predict
from fastapi.responses import HTMLResponse


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Retail Sales Prediction API",
    description="""
This API exposes a machine learning model trained to predict **weekly retail sales** based on operational and environmental variables.

### Model

The model is a Ridge regression trained on synthetic retail data simulating real-world conditions.

### Target variable

**weekly_sales**

Represents total weekly revenue for a product or store unit.

### Input features

- **price** — product price
- **promotion** — whether promotion is active (0 or 1)
- **temperature** — environmental temperature

### Prediction  

- Go to "POST /predict"
- "Try it out"   
- Input, for example:

{
  "price": 12.5,
  "promotion": 1,
  "temperature": 25
}

- Click on "Execute"

### Output

Predicted weekly sales value.

This project demonstrates a complete MLOps pipeline including training, packaging, CI/CD, Docker, and deployment.
""",
    version="1.0.0",
)


from pydantic import BaseModel, Field


class InputSchema(BaseModel):

    price: float = Field(
        description="Product price",
        example=12.5
    )

    promotion: int = Field(
        description="Promotion flag (0 = no promotion, 1 = promotion active)",
        example=1
    )

    temperature: float = Field(
        description="Environmental temperature",
        example=25
    )

    
@app.on_event("startup")
def startup_event():
    logger.info("API startup completed")


@app.get("/", response_class=HTMLResponse)
def root():

    return """
    <html>
        <head>
            <title>Retail Sales Prediction API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f4f6f8;
                    color: #333;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    max-width: 700px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    margin-top: 0;
                }
                a {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 15px;
                    background-color: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                a:hover {
                    background-color: #1e40af;
                }
                .meta {
                    margin-top: 20px;
                    font-size: 14px;
                    color: #666;
                }
            </style>
        </head>
        <body>

            <div class="container">

                <h1>Retail Sales Prediction API</h1>

                <p>
                Machine learning inference service deployed using FastAPI, Docker, and CI/CD.
                </p>

                <a href="/docs">Open interactive documentation/Execution</a>

                <div class="meta">

                    <p><strong>Model:</strong> Ridge regression</p>

                    <p><strong>Target:</strong> weekly retail sales</p>

                    <p><strong>Status:</strong> running</p>

                </div>

            </div>

        </body>
    </html>
    """



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
