import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

try:
    from utils.transform import get_random_transform_adaptive, apply_transform
except Exception:
    from transform import get_random_transform_adaptive, apply_transform


SUPPORTED_EXTS = ('.ply', '.pcd', '.xyz', '.txt', '.pts', '.npy', '.npz')


def farthest_point_sample(point, npoint):
    N, D = point.shape
    if N <= 0:
        raise ValueError("FPS: nuage vide")
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


def _find_files_recursive(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(SUPPORTED_EXTS):
                files.append(os.path.join(dirpath, f))
    files.sort()
    return files


def _load_points_from_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.ply', '.pcd'):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points).astype(np.float32)
        return pts

    if ext in ('.xyz', '.txt', '.pts'):
        pts = np.loadtxt(path, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] < 3:
            raise ValueError(f"{path}: besoin d'au moins 3 colonnes (x y z)")
        return pts[:, :3].astype(np.float32)

    if ext == '.npy':
        pts = np.asarray(np.load(path), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"{path}: attendu (N,3)")
        return pts[:, :3].astype(np.float32)

    if ext == '.npz':
        data = np.load(path)
        for k in ['points', 'xyz', 'pc', 'arr_0']:
            if k in data:
                pts = np.asarray(data[k], dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    return pts[:, :3].astype(np.float32)
        raise ValueError(f"{path}: aucune clé points/xyz/pc/arr_0 trouvée")

    raise ValueError(f"Extension non supportée: {ext}")


class CrossSourceDataset(Dataset):

    def __init__(self, root_dir, mode='train', num_points=1024):
        folder_name = 'train_data' if mode == 'train' else 'test_data'

        cand1 = os.path.join(root_dir, 'data', folder_name)
        cand2 = os.path.join(root_dir, folder_name)

        if os.path.exists(cand1):
            self.src_dir = cand1
        elif os.path.exists(cand2):
            self.src_dir = cand2
        else:
            self.src_dir = cand1  # pour afficher un message clair

        self.num_points = int(num_points)

        self.filepaths = _find_files_recursive(self.src_dir)
        self.data_cache = []

        # NO RESIZE: on centre seulement, scale=1
        self.norm_params = {}   # basename -> (mean, 1.0)
        self.path_map = {}      # basename -> path

        self.current_max_angle = np.deg2rad(30)

        print(f"[DATA] Chargement {mode.upper()} depuis {self.src_dir} ...")
        print(f"      {len(self.filepaths)} fichiers trouvés ({SUPPORTED_EXTS}).")

        for path in self.filepaths:
            fname = os.path.basename(path)
            try:
                points = _load_points_from_file(path)
                if points.shape[0] < 10:
                    continue

                self.path_map[fname] = path

                # downsample RAM (ne change pas l'échelle)
                if len(points) > 10000:
                    indices = np.random.choice(len(points), 10000, replace=False)
                    points = points[indices]

                # centre seulement (pas de /scale)
                mean = np.mean(points, axis=0).astype(np.float32)
                points = (points - mean).astype(np.float32)
                scale = 1.0

                self.norm_params[fname] = (mean, scale)
                self.data_cache.append((points, fname))

            except Exception as e:
                print(f"[WARN] Skip {fname}: {e}")

        if len(self.data_cache) == 0:
            print("\n[ERREUR] Dataset vide après chargement.")
            print("Vérifie que tes fichiers sont bien dans :")
            print(f"  - {cand1}")
            print(f"  - ou {cand2}\n")

        # duplication si dataset trop petit
        if len(self.data_cache) > 0 and len(self.data_cache) < 32:
            while len(self.data_cache) < 64:
                self.data_cache = self.data_cache + self.data_cache

    def set_difficulty(self, degrees):
        self.current_max_angle = np.deg2rad(degrees)

    def __len__(self):
        return len(self.data_cache)

    def get_norm_params(self, fname):
        return self.norm_params.get(fname, (np.zeros(3, dtype=np.float32), 1.0))

    def get_file_path(self, fname):
        return self.path_map.get(fname, os.path.join(self.src_dir, fname))

    def __getitem__(self, idx):
        points_original, fname = self.data_cache[idx]

        # target = FPS (downsample) mais échelle conservée
        pts_tgt = farthest_point_sample(points_original, self.num_points)

        # source = target transformée
        pts_src_clean = pts_tgt.copy()

        R, t = get_random_transform_adaptive(pts_tgt, self.current_max_angle)
        pts_src = apply_transform(pts_src_clean, R, t)

        # bruit (en unités du dataset)
        noise = np.random.normal(0, 0.005, pts_src.shape).astype(np.float32)
        pts_src += noise

        src_tensor = torch.from_numpy(pts_src).float().transpose(1, 0)  # (3,N)
        tgt_tensor = torch.from_numpy(pts_tgt).float().transpose(1, 0)  # (3,N)
        true_R = torch.from_numpy(R).float()
        true_t = torch.from_numpy(t).float()

        return src_tensor, tgt_tensor, true_R, true_t, fname