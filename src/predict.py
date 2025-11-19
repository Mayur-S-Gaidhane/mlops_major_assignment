"""Load sklearn model.joblib or quantized params and print R^2 on test set.
Used by CI and Docker container as the verification step.
"""
import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ART = os.path.join("artifacts")
model_path = os.path.join(ART, "model.joblib")

def main():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found. Run src/train.py first to create {model_path}")

    model = joblib.load(model_path)

    data = fetch_california_housing()
    X = data.data
    y = data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Verification: sklearn model Test R^2 = {r2:.6f}")

if __name__ == '__main__':
    main()
