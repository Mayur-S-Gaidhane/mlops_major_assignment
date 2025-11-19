# MLOps Major Assignment

Complete project scaffold for the MLOps major assignment (California Housing dataset, sklearn LinearRegression -> PyTorch single-layer model, manual uint8 quantization, Docker + CI).

**How this repo is organized**
```
/src
  train.py         # trains sklearn LinearRegression, saves model.joblib
  torch_model.py   # PyTorch single-layer model helper
  quantize.py      # manual uint8 quantize & dequantize, saves param files
  predict.py       # loads model.joblib, runs predictions and prints R^2
/requirements.txt
/environment.yml
/Dockerfile
/.github/workflows/ci.yml
/Makefile
/tests/verify.sh
/docs/              # templates for screenshots and final PDF text
```

## Quick local run (WSL + conda)

1. Create conda env:
```bash
conda env create -f environment.yml
conda activate mlops_major
conda install -y pytorch cpuonly -c pytorch
```

2. Train & generate files:
```bash
python src/train.py
```

3. Quantize (creates unquant_params.joblib and quant_params.joblib):
```bash
python src/quantize.py
```

4. Run prediction (prints R^2):
```bash
python src/predict.py
```

## Docker (build & run)
Build:
```bash
docker build -t <dockerhub_user>/mlops_major:dev .
```
Run:
```bash
docker run --rm <dockerhub_user>/mlops_major:dev
```

## Notes
- `fetch_california_housing` may download data on first run (internet required).
- The quantization is manual uint8 mapping with stored `scale` and `min` to dequantize.
- Followed the assignment's branching strategy when creating commits & branches.



---

## 📊 Quantization Analysis & Comparison Table

The following table fulfills the Step 4 requirement to compare the original
scikit-learn model and the quantized PyTorch model.

### **Model Performance & Size Comparison**

| **Metric**       | **Original Sklearn Model** | **Quantized Model** |
|------------------|----------------------------|----------------------|
| **R² Score**     | `0.575788`                 | `0.559582` (after fine-tune) |
| **Model Size**   | size of `unquant_params.joblib` = **0.369 KB** | size of `quant_params.joblib` = **0.386 KB** |

**Notes:**
- Model size increased slightly due to metadata storage in quantization.
- R² score remains close to original after 50-epoch fine-tuning.
- Quantized model is functional and stable.

