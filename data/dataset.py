
import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

try:
    from utils.transform import get_random_transform, apply_transform
except Exception:
    from transform import get_random_transform, apply_transform


def farthest_point_sample(point, npoint):

    # FPS : sélectionne les points les plus espacés pour capturer la structure (coins, murs).
 
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(N, npoint, replace=True)
        return point[indices]

    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((N,), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return point[centroids]


class CrossSourceDataset(Dataset):


    def __init__(self, root_dir, mode='train', num_points=1024):
        folder_name = 'train_data' if mode == 'train' else 'test_data'

        self.src_dir = os.path.join(root_dir, 'data', folder_name)
        if not os.path.exists(self.src_dir):
            self.src_dir = os.path.join(root_dir, folder_name)

        self.num_points = int(num_points)
        self.files = [f for f in os.listdir(self.src_dir) if f.endswith(('.ply', '.pcd'))]
        self.data_cache = []

        # Pour HD/UHD cohérent avec le training
        self.norm_params = {}   # fname -> (mean(3,), scale(float))
        self.path_map = {}      # fname -> chemin absolu

        # 30 degrés (visible mais gérable)
        self.current_max_angle = np.deg2rad(30)

        print(f"[DATA] Chargement {mode.upper()} depuis {self.src_dir} ...")
        print(f"      {len(self.files)} fichiers trouvés.")

        for fname in self.files:
            try:
                path = os.path.join(self.src_dir, fname)
                pcd = o3d.io.read_point_cloud(path)
                points = np.asarray(pcd.points).astype(np.float32)

                if len(points) < 10:
                    continue

                # cache path
                self.path_map[fname] = path

                # Downsampling RAM avant FPS
                if len(points) > 10000:
                    indices = np.random.choice(len(points), 10000, replace=False)
                    points = points[indices]

                # Normalisation training
                mean = np.mean(points, axis=0).astype(np.float32)
                points = points - mean
                max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1))).astype(np.float32)
                scale = float(max_dist) if max_dist > 0 else 1.0
                points = points / scale

                # stock norm params pour HD/UHD
                self.norm_params[fname] = (mean, scale)

                self.data_cache.append((points, fname))
            except Exception as e:
                print(f"Erreur {fname}: {e}")

        # Duplication si dataset trop petit
        if len(self.data_cache) > 0 and len(self.data_cache) < 32:
            while len(self.data_cache) < 64:
                self.data_cache = self.data_cache + self.data_cache

    def set_difficulty(self, degrees):
        self.current_max_angle = np.deg2rad(degrees)

    def __len__(self):
        return len(self.data_cache)

    def get_norm_params(self, fname):
        if fname in self.norm_params:
            return self.norm_params[fname]
        return np.zeros(3, dtype=np.float32), 1.0

    def get_file_path(self, fname):
        if fname in self.path_map:
            return self.path_map[fname]
        return os.path.join(self.src_dir, fname)

    def __getitem__(self, idx):
        points_original, fname = self.data_cache[idx]

        # Target = FPS
        pts_tgt = farthest_point_sample(points_original, self.num_points)

        # Source = même target transformé
        pts_src_clean = pts_tgt.copy()

        R, t = get_random_transform(self.current_max_angle)
        pts_src = apply_transform(pts_src_clean, R, t)

        # bruit
        noise = np.random.normal(0, 0.005, pts_src.shape).astype(np.float32)
        pts_src += noise

        src_tensor = torch.from_numpy(pts_src).float().transpose(1, 0)  # (3,N)
        tgt_tensor = torch.from_numpy(pts_tgt).float().transpose(1, 0)  # (3,N)
        true_R = torch.from_numpy(R).float()
        true_t = torch.from_numpy(t).float()

        return src_tensor, tgt_tensor, true_R, true_t, fname
