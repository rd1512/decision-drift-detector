from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to model artifacts
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

# Model expectations
NUM_FEATURES = 30

# Optional threshold (not enforced yet)
CONFIDENCE_THRESHOLD = 0.5
