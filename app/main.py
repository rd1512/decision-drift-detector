import time
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.inference import predict
from drift.collector import log_prediction


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
    start_time = time.time()

    try:
        decision, probability, input_summary = predict(request.features)

        latency_ms = (time.time() - start_time) * 1000

        prediction_id = log_prediction(
            decision=decision,
            probability=probability,
            latency_ms=latency_ms,
            input_summary=input_summary
        )

        return PredictionResponse(
            decision=decision,
            probability=probability,
            prediction_id=prediction_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
