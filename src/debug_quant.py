import os, joblib, numpy as np, torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.torch_model import build_from_sklearn_params

ART = "artifacts"
unq = joblib.load(os.path.join(ART, "unquant_params.joblib"))
q = joblib.load(os.path.join(ART, "quant_params.joblib"))

coef = np.asarray(unq['coef']).reshape(-1)
intercept = np.asarray(unq['intercept']).reshape(-1)

q_coef = np.asarray(q['q_coef']).reshape(-1)
q_intercept = np.asarray(q['q_intercept']).reshape(-1)
meta = q['meta']

def dequant_coef(qarr, meta):
    max_abs = meta['coef_max_abs']
    if max_abs == 0:
        return np.zeros_like(qarr, dtype=np.float64)
    return (qarr.astype(np.float64) / 255.0) * (2.0 * max_abs) - max_abs

def dequant_intercept(qarr, meta):
    max_abs = meta['intercept_max_abs']
    if max_abs == 0:
        return np.zeros_like(qarr, dtype=np.float64)
    return (qarr.astype(np.float64) / 255.0) * (2.0 * max_abs) - max_abs

decoef = dequant_coef(q_coef, meta).astype(np.float32).reshape(coef.shape)
deint = dequant_intercept(q_intercept, meta).astype(np.float32).reshape(intercept.shape)

print("=== PARAMS ===")
print("sklearn coef (first 8):", np.array2string(coef, precision=6))
print("dequant coef  (first 8):", np.array2string(decoef, precision=6))
print("coef max abs diff:", np.max(np.abs(coef - decoef)))
print("sklearn intercept:", intercept)
print("dequant intercept :", deint)
print("intercept abs diff:", np.max(np.abs(intercept - deint)))
print()

# Build models
skl = joblib.load(os.path.join(ART, "model.joblib"))
pt_model = build_from_sklearn_params(decoef, deint)
pt_state = pt_model.state_dict()

print("=== PYTORCH STATE DIAGNOSTICS ===")
print("fc.weight shape:", pt_state['fc.weight'].shape)
print("fc.bias shape  :", pt_state['fc.bias'].shape)
print("fc.weight (first row):", np.array2string(pt_state['fc.weight'].cpu().numpy().ravel()[:8], precision=6))
print("fc.bias:", np.array2string(pt_state['fc.bias'].cpu().numpy(), precision=6))
print()

# Load dataset and test
data = fetch_california_housing()
X = data.data; y = data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sklearn predictions
y_skl = skl.predict(X_test)

# PyTorch predictions
pt_model.eval()
with torch.no_grad():
    X_t = torch.from_numpy(X_test.astype(np.float32))
    y_pt = pt_model(X_t).cpu().numpy()

def stats(arr):
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "nan_count": int(np.isnan(arr).sum())
    }

print("=== PREDICTION SAMPLES (first 10) ===")
print("sklearn preds :", np.array2string(y_skl[:10], precision=6))
print("pytorch preds :", np.array2string(y_pt[:10], precision=6))
print()

print("=== PREDICTION STATS ===")
print("sklearn stats:", stats(y_skl))
print("pytorch  stats:", stats(y_pt))
print()

# Compute R2s
r2_skl = r2_score(y_test, y_skl)
r2_pt = r2_score(y_test, y_pt)
print(f"R2 sklearn: {r2_skl:.6f}")
print(f"R2 dequantized PyTorch: {r2_pt:.6f}")

# Extra: sample check
print()
print("=== SAMPLE CHECK (first test row) ===")
x0 = X_test[0].astype(np.float32)
print("x0 (first 8 features):", np.array2string(x0[:8], precision=6))
print("sklearn dot + intercept:", float(np.dot(coef, x0) + float(intercept)))
print("dequant dot + intercept :", float(np.dot(decoef, x0) + float(deint)))
print("pytorch single forward:", float(pt_model(torch.from_numpy(x0.reshape(1,-1))).item()))
