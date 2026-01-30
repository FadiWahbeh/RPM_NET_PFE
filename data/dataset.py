import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from utils.transform import get_random_transform, apply_transform

def farthest_point_sample(point, npoint):
    """ 
    FPS : Sélectionne les points les plus espacés pour capturer la structure (coins, murs).
    Indispensable pour que l'IA comprenne la forme avec peu de points.
    """
    N, D = point.shape
    if N < npoint:
        # Si pas assez de points, on complète aléatoirement
        indices = np.random.choice(N, npoint, replace=True)
        return point[indices]
        
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)
    return point[centroids.astype(np.int32)]

class CrossSourceDataset(Dataset):
    def __init__(self, root_dir, mode='train', num_points=1024):
        folder_name = 'train_data' if mode == 'train' else 'test_data'
        
        self.src_dir = os.path.join(root_dir, 'data', folder_name)
        if not os.path.exists(self.src_dir):
             self.src_dir = os.path.join(root_dir, folder_name)

        self.num_points = num_points
        self.files = [f for f in os.listdir(self.src_dir) if f.endswith(('.ply', '.pcd'))]
        self.data_cache = []
        
        # --- MODIFICATION VISUELLE ---
        # 30 degrés : Suffisant pour être vu à l'œil nu, mais gérable pour l'IA.
        self.current_max_angle = np.deg2rad(30) 

        print(f"[DATA] Chargement {mode.upper()} depuis {self.src_dir} ...")
        for fname in self.files:
            try:
                path = os.path.join(self.src_dir, fname)
                pcd = o3d.io.read_point_cloud(path)
                points = np.asarray(pcd.points)
                
                # Downsampling de sécurité (RAM) avant FPS
                if len(points) > 10000:
                    indices = np.random.choice(len(points), 10000, replace=False)
                    points = points[indices]
                
                # Normalisation (Centrage + Échelle unitaire)
                points = points - np.mean(points, axis=0)
                max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
                if max_dist > 0:
                    points = points / max_dist
                    
                self.data_cache.append((points, fname))
            except Exception as e: print(f"Erreur {fname}: {e}")
        
        # Duplication si dataset trop petit
        if len(self.data_cache) > 0 and len(self.data_cache) < 32:
            while len(self.data_cache) < 64:
                self.data_cache = self.data_cache + self.data_cache
        
    def set_difficulty(self, degrees):
        self.current_max_angle = np.deg2rad(degrees)

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        points_original, fname = self.data_cache[idx]
        
        # 1. Sélection intelligente des points (Target)
        pts_tgt = farthest_point_sample(points_original, self.num_points)
        
        pts_src_clean = pts_tgt.copy()

        # 2. Application de la transformation visible (Source)
        R, t = get_random_transform(self.current_max_angle)
        pts_src = apply_transform(pts_src_clean, R, t)
        
        # 3. Ajout léger de bruit
        noise = np.random.normal(0, 0.005, pts_src.shape)
        pts_src += noise

        # 4. Conversion Tensor
        src_tensor = torch.from_numpy(pts_src).float().transpose(1, 0)
        tgt_tensor = torch.from_numpy(pts_tgt).float().transpose(1, 0)
        true_R = torch.from_numpy(R).float()
        true_t = torch.from_numpy(t).float()
        
        return src_tensor, tgt_tensor, true_R, true_t, fname