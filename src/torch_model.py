"""Single-layer PyTorch model and helpers to set weights from sklearn params."""
import torch
import torch.nn as nn
import numpy as np

class SingleLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1, bias=True)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

def build_from_sklearn_params(sklearn_coef, sklearn_intercept, device='cpu', dtype=torch.float32):
    """Create SingleLinear and load weights from sklearn params (numpy arrays)."""
    n = int(sklearn_coef.shape[-1])
    model = SingleLinear(n)
    # set weights: PyTorch weight shape is (out_features, in_features)
    coef = np.asarray(sklearn_coef).reshape(1, n)
    intercept = np.asarray(sklearn_intercept).reshape(1)
    with torch.no_grad():
        model.fc.weight.data = torch.from_numpy(coef).to(dtype)
        model.fc.bias.data = torch.from_numpy(intercept).to(dtype)
    return model
