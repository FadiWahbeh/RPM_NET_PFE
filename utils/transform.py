import numpy as np
import torch

# Rotation 3D à partir d'Euler XYZ.
def _rotation_from_euler(anglex: float, angley: float, anglez: float) -> np.ndarray:
    cosx = np.cos(anglex); sinx = np.sin(anglex)
    cosy = np.cos(angley); siny = np.sin(angley)
    cosz = np.cos(anglez); sinz = np.sin(anglez)

    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]], dtype=np.float32)

    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]], dtype=np.float32)

    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)

# Taille caractéristique = diagonale de l'AABB.
def estimate_object_size(points_xyz: np.ndarray) -> float:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("estimate_object_size: points_xyz doit être (N,3)")
    pmin = points_xyz.min(axis=0)
    pmax = points_xyz.max(axis=0)
    diag = float(np.linalg.norm(pmax - pmin))
    return diag if diag > 1e-9 else 1.0

"""
Rotation + translation adaptatives: distance = facteur * taille_objet.
source/target automatiquement séparés selon la taille réelle.
"""
def get_random_transform_adaptive(points_xyz: np.ndarray,
                                 max_angle_radians: float,
                                 min_sep_factor: float = 1.25,
                                 max_sep_factor: float = 2.25,
                                 jitter_factor: float = 0.05):

    size = estimate_object_size(points_xyz)

    # Rotation
    anglex = np.random.uniform(-max_angle_radians, max_angle_radians)
    angley = np.random.uniform(-max_angle_radians, max_angle_radians)
    anglez = np.random.uniform(-max_angle_radians, max_angle_radians)
    R = _rotation_from_euler(anglex, angley, anglez)

    # Direction + distance proportionnelle
    direction = np.random.normal(size=3).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = np.array([1, 0, 0], dtype=np.float32)
        norm = 1.0
    direction /= norm

    base_dist = float(np.random.uniform(min_sep_factor, max_sep_factor) * size)
    jitter = float(np.random.uniform(-jitter_factor, jitter_factor) * size)
    distance = max(0.0, base_dist + jitter)
    t = direction * distance

    return R.astype(np.float32), t.astype(np.float32)


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t


def transform_point_cloud_torch(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # points: (B,3,N), R: (B,3,3), t: (B,3)
    rotated = torch.bmm(R, points)
    translated = rotated + t.unsqueeze(2)
    return translated


# Transforme une matrice de scores (B,N,N) en une matrice de probabilités.
def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    my_log_alpha = log_alpha.clone()
    for _ in range(n_iters):
        my_log_alpha = my_log_alpha - torch.logsumexp(my_log_alpha, dim=2, keepdim=True)
        my_log_alpha = my_log_alpha - torch.logsumexp(my_log_alpha, dim=1, keepdim=True)
    return torch.exp(my_log_alpha)