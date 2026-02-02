from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Input features in the same order as the training data"
    )


class PredictionResponse(BaseModel):
    decision: str
    probability: float
    prediction_id: str
