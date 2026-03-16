import numpy as np

import torch

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))


def rmse_GPU(y_true, y_pred):
    # Sécurité : on force CPU + detach
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu()

    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
