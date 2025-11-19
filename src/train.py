"""
Train sklearn LinearRegression on California Housing and save model.joblib and a PyTorch .pth model

Saves:
- artifacts/model.joblib         (sklearn model)
- artifacts/model.pth            (PyTorch single-layer model state_dict)
- artifacts/unquant_params.joblib (float params saved; used by quantize.py)
"""
import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import torch helper to build a single-layer model
from src.torch_model import build_from_sklearn_params
import torch

ARTIFACT_DIR = os.path.join("artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    # Load dataset (may download on first run)
    data = fetch_california_housing()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Trained LinearRegression. Test R^2 = {r2:.6f}")

    # Save sklearn model
    model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved sklearn model -> {model_path}")

    # Save raw parameters for quantization (float arrays)
    params = {
        'coef': model.coef_.astype(float),
        'intercept': np.array([model.intercept_]).astype(float)
    }
    unquant_path = os.path.join(ARTIFACT_DIR, "unquant_params.joblib")
    joblib.dump(params, unquant_path)
    print(f"Saved unquantized params -> {unquant_path}")

    # Build a PyTorch single-layer model and save its state_dict as a .pth file
    try:
        # build_from_sklearn_params expects coef shape compatible; pass coef and intercept
        torch_model = build_from_sklearn_params(model.coef_, np.array([model.intercept_]))
        pth_path = os.path.join(ARTIFACT_DIR, "model.pth")
        torch.save(torch_model.state_dict(), pth_path)
        print(f"Saved PyTorch model state_dict -> {pth_path}")
    except Exception as e:
        print("Warning: Failed to build/save PyTorch .pth model:", e)

if __name__ == '__main__':
    main()
