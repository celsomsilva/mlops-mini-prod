# mlops-mini-prod

A minimal, end-to-end **MLOps project** designed to practice taking a Machine Learning model from experimentation to production.

The goal here is not to build the smartest model, but to build a model that can actually **run in production**.

This repository shows the complete lifecycle: train, package, serve, containerize, test, CI and deploy-ready

I chose scikit-learn here because the focus is MLOps, not model complexity.

---

---

## Project Status

This repository is **work-in-progress**.  

---

---

## Purpose

Many ML projects stop at notebooks.

This one focuses on what companies actually expect in real-world systems:

- reproducible training
- saved model artifacts
- inference API
- Docker container
- automated tests
- CI pipeline

Think of it as a **production-ready template** for future ML systems.

---

## Full workflow

1. Train model  
  
2. Save artifact  
  
3. API loads model  
  
4. Docker container  
  
5. Tests + CI validation  
  
6. Ready to deploy  

---

## Project structure

```
mlops-mini-prod/
  src/
   mlops_api/
        __init__.py
   	api.py			#FastAPI application (/health, /predict)
   	train.py			#training script that generates the model     
   	predict.py			#inference logic
  
  models/			#saved artifacts (model.joblib + metadata.json)
  
  tests/			#automated tests using pytest
  
  Dockerfile			#application container
  compose.yaml			#local docker execution
  .github/workflows/ci.yaml		# CI pipeline (GitHub Actions)
  Makefile			#shortcut commands
  requirements.txt		#dependencies
  pyproject.toml 		# package configuration (src-layout)
```

> The project follows the **src-layout packaging pattern**, so the application code is installed as a Python package (`mlops_api`).



---

## Running locally(development mode)

Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate


Install dependencies and the local package
pip install -r requirements.txt
pip install -e .

Train the model(offline step)
make train

Run tests
make test

Start the API
make run

Open:
http://localhost:8000/docs

Interactive Swagger UI is available automatically.

---

## Running with Docker (production-like environment)

make docker

or

docker compose up --build

---

## Running tests

make test

---

## CI/CD (GitHub Actions)

On every push or pull request the pipeline automatically:

- checks out the repository
- sets up a clean Python environment
- installs dependencies
- runs tests
- builds Docker image
- fails fast if something breaks

This guarantees the project is always deployable.

---

## API Endpoints

### Health check 
 
GET /health  

Response: 

{
  "status": "ok"
	}

### Prediction  

POST /predict  

Input:

{
  "price": 12.5,
  "promotion": 1,
  "temperature": 25
}


Output:

{
  "prediction": 180.3,
  "model_version": "2026-02-05T18:12:00",
  "rmse": 10.63
}


---

## Tech stack

- Python 3.11
- FastAPI
- scikit-learn
- Docker
- pytest
- GitHub Actions

---

## Why this project exists

This project was created as a practical exercise to bridge the gap between:

> “Model works on my notebook”
> and
> “Model runs reliably in production”.

It intentionally keeps the ML simple and focuses on engineering best practices.

After this template is in place, it can easily be reused for:

- forecasting systems
- recommendation engines
- NLP / LLM services
- RAG pipelines
- any ML microservice

---

## Possible extensions

- MLflow model registry
- structured logging
- Prometheus metrics
- automatic deployment (Render / Fly.io / Railway)
- batch inference
- LLM integration

---

## Author

This project was developed by an engineer and data scientist with a background in:

* Postgraduate degree in **Data Science and Analytics (USP)**
* Bachelor's degree in **Computer Engineering (UERJ)**
* Special interest in statistical models, interpretability, and applied AI
* Strong interest in algorithmic reasoning, correctness, and performance evaluation

---

## Contact

* [LinkedIn](https://linkedin.com/in/celso-m-silva)
* Or open an [issue](https://github.com/celsomsilva/mlops-mini-prod/issues)
