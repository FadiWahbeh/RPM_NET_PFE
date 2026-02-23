import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


def _to_numpy_xy(source, target):
    if torch.is_tensor(source):
        source = source.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # accepte (3,N) ou (N,3)
    if source.shape[0] == 3:
        source = source.T
    if target.shape[0] == 3:
        target = target.T

    return source.astype(np.float32), target.astype(np.float32)


def _diag_aabb(points_np: np.ndarray) -> float:
    pmin = points_np.min(axis=0)
    pmax = points_np.max(axis=0)
    diag = float(np.linalg.norm(pmax - pmin))
    return max(diag, 1e-9)


def compute_rmse_cloudcompare(source, target) -> float:
    source, target = _to_numpy_xy(source, target)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target)
    distances, _ = nbrs.kneighbors(source)

    rmse = float(np.sqrt(np.mean(distances ** 2)))
    return rmse


def compute_rmse_normalized(source, target) -> float:
    """
    RMSE normalisé (sans unité) : rmse / diag(AABB_target)
    -> comparable entre objets de tailles différentes.
    """
    source, target = _to_numpy_xy(source, target)
    rmse = compute_rmse_cloudcompare(source, target)
    diag = _diag_aabb(target)
    return float(rmse / diag)


def compute_lcp_adaptive(source, target, ratio=0.02) -> float:
    """
    LCP adaptatif: threshold = ratio * diag(AABB_target)
    Exemple: ratio=0.02 => 2% de la taille de l'objet.
    """
    source, target = _to_numpy_xy(source, target)

    diag = _diag_aabb(target)
    thresh = max(1e-9, float(ratio * diag))

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target)
    distances, _ = nbrs.kneighbors(source)
    return float(np.mean(distances.ravel() < thresh))