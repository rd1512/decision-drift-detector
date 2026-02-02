import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset (for demo inference)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Random real sample from dataset
sample_input = X.sample(n=1).values

# Apply same preprocessing
sample_input_scaled = scaler.transform(sample_input)

# Inference
prediction = model.predict(sample_input_scaled)
probability = model.predict_proba(sample_input_scaled)

print("Prediction:", prediction[0])
print("Confidence:", probability.max())
