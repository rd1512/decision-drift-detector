import joblib
import numpy as np

from app.config import MODEL_PATH, SCALER_PATH, NUM_FEATURES


# Load model and scaler once at startup
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict(features: list[float]) -> tuple[str, float]:
    """
    Run model inference and return decision and probability
    """

    if len(features) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {len(features)}")

    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled).max()

    decision = "benign" if prediction == 1 else "malignant"

    return decision, round(float(probability), 4)
