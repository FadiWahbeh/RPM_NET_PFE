import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


def _to_numpy_Nx3(pts) -> np.ndarray:
    """
    Convertit tensor ou array en (N,3) numpy float32.
    ✅ FIX shape: double condition — évite la transposition erronée si N==3.
    L'ancienne version faisait shape[0]==3 sans vérifier shape[1],
    ce qui transposait incorrectement un nuage de 3 points en (3,3).
    """
    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    return pts


def _build_nn(target: np.ndarray) -> NearestNeighbors:
    """
    Construit et retourne un KD-tree fitté sur target.
    ✅ Séparé pour clarté et réutilisation potentielle.
    algorithm='kd_tree' est plus rapide que 'auto' pour des nuages 3D denses.
    """
    return NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)


def compute_rmse_cloudcompare(source, target) -> float:
    """
    RMSE symétrique entre deux nuages de points.

    Formule: sqrt( (MSE(src→tgt) + MSE(tgt→src)) / 2 )

    ✅ FIX asymétrie: l'ancienne version ne calculait que src→tgt.
    Un RMSE one-way peut être faible même si la cible a des zones
    entières non couvertes par la source (faux positif d'alignement).
    La version symétrique pénalise les deux sens.

    ✅ FIX robustesse: garde contre nuages vides.
    """
    source = _to_numpy_Nx3(source)
    target = _to_numpy_Nx3(target)

    if source.shape[0] == 0 or target.shape[0] == 0:
        return 0.0

    nn_tgt = _build_nn(target)
    nn_src = _build_nn(source)

    d_s2t, _ = nn_tgt.kneighbors(source)   # (N,1) distances Euclidiennes src→tgt
    d_t2s, _ = nn_src.kneighbors(target)   # (M,1) distances Euclidiennes tgt→src

    mse_s2t = np.mean(d_s2t ** 2)
    mse_t2s = np.mean(d_t2s ** 2)

    return float(np.sqrt((mse_s2t + mse_t2s) / 2.0))


def compute_lcp_adaptive(source, target, ratio: float = 0.02) -> float:
    """
    LCP adaptatif: threshold = ratio * diag(AABB_target).
    ratio=0.02 => 2% de la taille de l'objet cible.

    ✅ FIX robustesse: garde contre nuages vides.
    """
    source = _to_numpy_Nx3(source)
    target = _to_numpy_Nx3(target)

    if source.shape[0] == 0 or target.shape[0] == 0:
        return 0.0

    diag = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    thresh = max(1e-9, ratio * diag)

    nn_tgt = _build_nn(target)
    distances, _ = nn_tgt.kneighbors(source)
    return float(np.mean(distances.ravel() < thresh))
