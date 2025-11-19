import os
import joblib
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.torch_model import build_from_sklearn_params

ART = os.path.join("artifacts")
os.makedirs(ART, exist_ok=True)

def quantize_symmetric_uint8_single(arr):
    arr = np.asarray(arr, dtype=np.float64)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0:
        q = np.zeros_like(arr, dtype=np.uint8)
    else:
        q = np.round((arr + max_abs) / (2.0 * max_abs) * 255.0).astype(np.uint8)
    meta = {'max_abs': max_abs}
    return q, meta

def dequantize_symmetric_uint8_single(q, meta):
    max_abs = meta['max_abs']
    if max_abs == 0:
        return np.zeros_like(q, dtype=np.float64)
    return (q.astype(np.float64) / 255.0) * (2.0 * max_abs) - max_abs

def finetune_model(model, X_train, y_train, epochs=50, lr=1e-4, clip_norm=1.0):
    # model: PyTorch model with linear fc layer
    model.train()
    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32)).squeeze()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    loss_fn = torch.nn.MSELoss()
    # save pre-finetune state
    pre_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        opt.zero_grad()
        preds = model(X_t).squeeze()              # ensure preds shape (N,)
        loss = loss_fn(preds, y_t)
        loss.backward()
        # gradient clipping to avoid explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        # quick NaN/Inf check in params after step
        bad = False
        for p in model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                bad = True
                break
        if bad:
            print(f"NaN/Inf detected in parameters at epoch {epoch}, reverting to pre-finetune state.")
            # revert
            model.load_state_dict(pre_state)
            return model, False
    # final check on outputs
    model.eval()
    with torch.no_grad():
        y_pred = model(X_t).squeeze().cpu().numpy()
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        print("NaN/Inf detected in predictions after finetune, reverting to pre-finetune state.")
        model.load_state_dict(pre_state)
        return model, False
    return model, True

def main(finetune=True, ft_epochs=50, ft_lr=1e-4):
    # Load unquantized params (created by train.py)
    unquant_path = os.path.join(ART, "unquant_params.joblib")
    if not os.path.exists(unquant_path):
        raise FileNotFoundError(f"Run src/train.py first to create {unquant_path}")
    params = joblib.load(unquant_path)
    coef = np.asarray(params['coef']).reshape(-1)
    intercept = np.asarray(params['intercept']).reshape(-1)

    # Save size info for unquantized
    size_unquant = os.path.getsize(unquant_path) / 1024.0

    # Quantize coef and intercept separately (symmetric uint8)
    q_coef, meta_coef = quantize_symmetric_uint8_single(coef)
    q_intercept, meta_intercept = quantize_symmetric_uint8_single(intercept)

    quant_params = {
        'q_coef': q_coef,
        'q_intercept': q_intercept,
        'meta': {
            'coef_max_abs': meta_coef['max_abs'],
            'intercept_max_abs': meta_intercept['max_abs']
        }
    }
    quant_path = os.path.join(ART, "quant_params.joblib")
    joblib.dump(quant_params, quant_path)
    size_quant = os.path.getsize(quant_path) / 1024.0

    print(f"Saved quantized params -> {quant_path}")
    print(f"Size unquant (KB): {size_unquant:.6f}")
    print(f"Size quant   (KB): {size_quant:.6f}")

    # Dequantize into float32
    decoef = dequantize_symmetric_uint8_single(q_coef, {'max_abs': quant_params['meta']['coef_max_abs']}).astype(np.float32).reshape(coef.shape)
    deint = dequantize_symmetric_uint8_single(q_intercept, {'max_abs': quant_params['meta']['intercept_max_abs']}).astype(np.float32).reshape(intercept.shape)

    # Diagnostics
    max_err_coef = np.max(np.abs(coef - decoef))
    max_err_intercept = np.max(np.abs(intercept - deint))
    print(f"Max abs err coef: {max_err_coef:.6e}")
    print(f"Max abs err intercept: {max_err_intercept:.6e}")

    # Build model & save pre-finetune state
    model = build_from_sklearn_params(decoef, deint)
    torch.save(model.state_dict(), os.path.join(ART, "quant_model_dequant_before_ft.pth"))

    # Load data and split
    data = fetch_california_housing()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate before fine-tune
    model.eval()
    with torch.no_grad():
        X_tst = torch.from_numpy(X_test.astype(np.float32))
        y_pred_before = model(X_tst).squeeze().cpu().numpy()
    r2_before = r2_score(y_test, y_pred_before)
    print(f"R^2 using dequantized PyTorch model (before fine-tune): {r2_before:.6f}")

    # Fine-tune
    success = True
    if finetune:
        print(f"Fine-tuning dequantized model for {ft_epochs} epochs (lr={ft_lr}) ...")
        model, success = finetune_model(model, X_train, y_train, epochs=ft_epochs, lr=ft_lr)
        if success:
            torch.save(model.state_dict(), os.path.join(ART, "quant_model_dequant.pth"))
            with torch.no_grad():
                y_pred_after = model(X_tst).squeeze().cpu().numpy()
            r2_after = r2_score(y_test, y_pred_after)
            print(f"R^2 using dequantized PyTorch model (after fine-tune): {r2_after:.6f}")
        else:
            # revert saved pre-ft state as final
            os.replace(os.path.join(ART, "quant_model_dequant_before_ft.pth"), os.path.join(ART, "quant_model_dequant.pth"))
            with torch.no_grad():
                y_pred_after = model(X_tst).squeeze().cpu().numpy()
            r2_after = r2_score(y_test, y_pred_after)
            print("Fine-tune failed and was reverted; reporting R2 using reverted dequantized model.")
            print(f"R^2 using dequantized PyTorch model (after revert): {r2_after:.6f}")
    else:
        os.replace(os.path.join(ART, "quant_model_dequant_before_ft.pth"), os.path.join(ART, "quant_model_dequant.pth"))
        r2_after = r2_before

    # Original sklearn R2
    skl_r2 = None
    skl_model_path = os.path.join(ART, "model.joblib")
    if os.path.exists(skl_model_path):
        skl = joblib.load(skl_model_path)
        y_pred_skl = skl.predict(X_test)
        skl_r2 = r2_score(y_test, y_pred_skl)

    print(f"Original sklearn model R^2 (for comparison): {skl_r2:.6f}" if skl_r2 is not None else "Original sklearn model missing.")
    return r2_before, r2_after, skl_r2

if __name__ == '__main__':
    main(finetune=True, ft_epochs=50, ft_lr=1e-4)
