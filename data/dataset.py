import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from utils.transform import get_random_transform, apply_transform

class SingleSourceDataset(Dataset):
    def __init__(self, root_dir, num_points=1024):
        self.src_dir = os.path.join(root_dir, 'data', 'source')
<<<<<<< HEAD
=======
        # Fallback pour trouver le dossier
>>>>>>> origin/main
        if not os.path.exists(self.src_dir):
             self.src_dir = os.path.join(root_dir, 'source')

        self.num_points = num_points
        self.files = [f for f in os.listdir(self.src_dir) if f.endswith(('.ply', '.pcd'))]
        self.data_cache = []
<<<<<<< HEAD
=======
        
        # On commence facile (45 degrés)
>>>>>>> origin/main
        self.current_max_angle = np.deg2rad(45) 

        print(f"[DATA] Chargement et Normalisation des fichiers...")
        for fname in self.files:
            try:
                path = os.path.join(self.src_dir, fname)
                pcd = o3d.io.read_point_cloud(path)
                points = np.asarray(pcd.points)
                
<<<<<<< HEAD
=======
                # --- CORRECTION CRUCIALE : NORMALISATION ---
>>>>>>> origin/main
                # 1. On centre l'objet en (0,0,0)
                points = points - np.mean(points, axis=0)
                # 2. On calcule sa taille maximale
                max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
                # 3. On le réduit pour qu'il tienne dans une sphère de rayon 1
                if max_dist > 0:
                    points = points / max_dist
                    
                self.data_cache.append((points, fname))
            except Exception as e: print(f"Erreur {fname}: {e}")
        
        # Duplication si pas assez de données
        if len(self.data_cache) > 0 and len(self.data_cache) < 32:
            while len(self.data_cache) < 64:
                self.data_cache = self.data_cache + self.data_cache
        
    def set_difficulty(self, degrees):
        self.current_max_angle = np.deg2rad(degrees)

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        points_original, fname = self.data_cache[idx]
        
        # Sampling
        if len(points_original) >= self.num_points:
            idx_rand = np.random.choice(len(points_original), self.num_points, replace=False)
        else:
            idx_rand = np.random.choice(len(points_original), self.num_points, replace=True)
            
        # DÉFINITION CLAIRE :
        # Target (Cible) = L'objet original centré en (0,0,0). IL EST STATIQUE.
        pts_tgt = points_original[idx_rand, :].copy()
        
        # Source = L'objet qu'on déplace loin et qu'on tourne.
        pts_src_clean = pts_tgt.copy()

        # Transformation Aléatoire
        R, t = get_random_transform(self.current_max_angle)
        
        # On applique la transfo à la SOURCE pour l'éloigner de la cible
        pts_src = apply_transform(pts_src_clean, R, t)
        
        # Ajout de bruit léger sur la source
        noise = np.random.normal(0, 0.005, pts_src.shape)
        pts_src += noise

        # Conversion PyTorch
        src_tensor = torch.from_numpy(pts_src).float().transpose(1, 0)
        tgt_tensor = torch.from_numpy(pts_tgt).float().transpose(1, 0)
        

        # On renvoie l'inverse de la transformation
        # Mais pour RPM, on lui donne juste les deux nuages.
        true_R = torch.from_numpy(R).float()
        true_t = torch.from_numpy(t).float()
        
        return src_tensor, tgt_tensor, true_R, true_t, fname