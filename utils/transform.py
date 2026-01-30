import numpy as np
import torch

def get_random_transform(max_angle_radians):
    anglex = np.random.uniform(-max_angle_radians, max_angle_radians)
    angley = np.random.uniform(-max_angle_radians, max_angle_radians)
    anglez = np.random.uniform(-max_angle_radians, max_angle_radians)
    
    cosx = np.cos(anglex); sinx = np.sin(anglex)
    cosy = np.cos(angley); siny = np.sin(angley)
    cosz = np.cos(anglez); sinz = np.sin(anglez)
    
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    # --- DISTANCE AUGMENTÉE (Séparation nette) ---
    # 0.5 à 1.0 : L'objet source est repoussé à une distance égale à 
    # la moitié ou la totalité de sa propre taille. Impossible qu'ils se chevauchent.
    distance = np.random.uniform(0.5, 1.0)
    
    direction = np.random.normal(size=3)
    norm = np.linalg.norm(direction)
    if norm == 0: norm = 1
    direction /= norm
    t = direction * distance
    
    return R, t

def apply_transform(points, R, t):
    return points @ R.T + t

def transform_point_cloud_torch(points, R, t):
    rotated = torch.bmm(R, points)
    translated = rotated + t.unsqueeze(2)
    return translated

def sinkhorn(log_alpha, n_iters=5):
    my_log_alpha = log_alpha.clone()
    for _ in range(n_iters):
        my_log_alpha = my_log_alpha - torch.logsumexp(my_log_alpha, dim=2, keepdim=True)
        my_log_alpha = my_log_alpha - torch.logsumexp(my_log_alpha, dim=1, keepdim=True)
    return torch.exp(my_log_alpha)