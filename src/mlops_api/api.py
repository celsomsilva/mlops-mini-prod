"""
FastAPI application.

Exposes health check and prediction endpoints for the ML service.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import logging
from mlops_api.predict import predict
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles



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

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


from pydantic import BaseModel, Field

class InputSchema(BaseModel):
    price: float = Field(
        gt=0,
        description="Product price in USD",
        example=12.5
    )
    promotion: int = Field(
        ge=0,
        le=1,
        description="Promotion flag (0 = no promotion, 1 = promotion active)",
        example=1
    )
    temperature: float = Field(
        description="Environmental temperature in Celsius",
        example=25
    )

    
@app.on_event("startup")
def startup_event():
    logger.info("API startup completed")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model": "Ridge regression",
            "target": "weekly retail sales",
            "status": "running"
        }
    )


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
