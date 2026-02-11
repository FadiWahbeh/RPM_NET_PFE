import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

def compute_rmse_cloudcompare(source, target):
    # Calcule le RMSE géométrique en trouvant le voisin le plus proche.
    
    if torch.is_tensor(source): source = source.detach().cpu().numpy()
    if torch.is_tensor(target): target = target.detach().cpu().numpy()
    
    if source.shape[0] == 3: source = source.T
    if target.shape[0] == 3: target = target.T

    # On utilise NearestNeighbors (KDTree)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target)
    distances, _ = nbrs.kneighbors(source)
    
    # Racine de la moyenne des carrés des distances
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

def compute_lcp_sklearn(source, target, threshold=0.05):
    if torch.is_tensor(source): source = source.detach().cpu().numpy()
    if torch.is_tensor(target): target = target.detach().cpu().numpy()
    
    if source.shape[0] == 3: source = source.T
    if target.shape[0] == 3: target = target.T

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target)
    distances, indices = nbrs.kneighbors(source)
    
    matches = distances.ravel() < threshold
    lcp_score = np.mean(matches)
    return lcp_score