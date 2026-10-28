# Heart Disease Risk Prediction API

A FastAPI service that serves a scikit-learn heart disease risk classifier.

## Requirements

- Python 3.11+
- Docker (optional)

## Run locally

From this directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive OpenAPI
documentation is available at `http://localhost:8000/docs`.

## Predict

Send a `POST` request to `/predict` with the five features used by the model:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"age":55,"trestbps":130,"chol":240,"thalach":150,"oldpeak":1.5}'
```

The response contains a numeric `prediction` (`0` or `1`) and a human-readable
`result`. Input values are validated before they reach the model.

## Test

```powershell
python -m pytest -q
```

## Docker

```powershell
docker build -t heart-api .
docker run --rm -p 8000:8000 heart-api
```

The container runs as a non-root user. The root endpoint (`/`) reports whether
the model loaded successfully.