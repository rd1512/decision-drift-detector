import os
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_data():
    #Load breast cancer dataset and return features and target
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def split_data(X, y):
    #Split data into train and test sets
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def scale_features(X_train, X_test):
    #Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train):
    #Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, X_test_scaled, y_test):
    #Evaluate model performance
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.5f}")


def save_artifacts(model, scaler):
    #Save trained model and scaler

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model and scaler saved successfully.")


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)
    save_artifacts(model, scaler)


if __name__ == "__main__":
    main()
