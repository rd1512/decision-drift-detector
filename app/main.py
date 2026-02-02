from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.inference import predict


app = FastAPI(
    title="Decision Drift Detector API",
    description="FastAPI service for ML model inference",
    version="1.0.0"
)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def run_prediction(request: PredictionRequest):
    try:
        decision, probability = predict(request.features)
        return PredictionResponse(
            decision=decision,
            probability=probability
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
