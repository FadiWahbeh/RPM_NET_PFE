import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

try:
    from utils.transform import apply_transform
except Exception:
    from transform import apply_transform


SUPPORTED_EXTS = ('.ply', '.pcd', '.xyz', '.txt', '.pts', '.npy', '.npz')


def _find_files(dir_path):
    if not os.path.exists(dir_path):
        return {}
    out = {}
    for f in os.listdir(dir_path):
        if f.lower().endswith(SUPPORTED_EXTS):
            out[f] = os.path.join(dir_path, f)
    return out


def _load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.ply', '.pcd'):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points).astype(np.float32)

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


def _aabb_diag(pts: np.ndarray) -> float:
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    d = float(np.linalg.norm(pmax - pmin))
    return d if d > 1e-9 else 1.0


def _center_only(pts: np.ndarray):
    mean = np.mean(pts, axis=0).astype(np.float32)
    return (pts - mean).astype(np.float32), mean


def _voxel_downsample_np(pts: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    return np.asarray(pcd.points).astype(np.float32)


def _statistical_outlier_removal_np(pts: np.ndarray, nb_neighbors=20, std_ratio=2.0) -> np.ndarray:
    if pts.shape[0] < 50:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return np.asarray(pcd.points).astype(np.float32)


def _farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """
    FPS vectorisé numpy.
    point: (N,3) → return: (npoint,3)

    ✅ FIX performance: np.minimum() remplace le masque booléen.
    L'ancienne version effectuait distance[mask] = dist[mask] avec
    une allocation d'un masque bool (N,) à chaque itération.
    np.minimum(distance, dist, out=distance) est équivalent mais
    ~2x plus rapide car opération in-place sans masque temporaire.
    """
    N = point.shape[0]
    if N <= 0:
        raise ValueError("FPS: nuage vide")
    if N < npoint:
        return point[np.random.choice(N, npoint, replace=True)]

    xyz = point[:, :3].astype(np.float32)
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.full(N, 1e10, dtype=np.float32)
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        # ✅ np.minimum in-place: pas de masque temporaire
        np.minimum(distance, dist, out=distance)
        farthest = int(np.argmax(distance))

    return point[centroids]


def _random_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    N = point.shape[0]
    if N <= 0:
        raise ValueError("Random sample: nuage vide")
    if N >= npoint:
        idx = np.random.choice(N, npoint, replace=False)
    else:
        idx = np.random.choice(N, npoint, replace=True)
    return point[idx]


def _random_rotation(max_angle_rad: float) -> np.ndarray:
    ax = np.random.uniform(-max_angle_rad, max_angle_rad)
    ay = np.random.uniform(-max_angle_rad, max_angle_rad)
    az = np.random.uniform(-max_angle_rad, max_angle_rad)

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


class PairedKinectLidarDataset(Dataset):
    """
    data/train_data/kinect + data/train_data/lidar
    data/test_data/kinect  + data/test_data/lidar

    Optimisation:
      - cache RAM des nuages preprocessés
      - pas d'Open3D read/filters à chaque __getitem__
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        num_points: int = 1024,
        # preprocess
        kinect_sor: bool = True,
        lidar_voxel: float = 0.05,
        lidar_sor: bool = True,
        # sampling
        src_sampling: str = "random",
        tgt_sampling: str = "fps",
        # separation
        sep_min_factor = 0.5,
        sep_max_factor = 1.0,
        # noise
        src_jitter_ratio: float = 0.002,
        # rotation
        max_rot_deg: float = 30.0,
        # cache control
        max_cache_points: int = 20000,
        preload_cache: bool = True,
    ):
        assert mode in ("train", "test")
        self.root_dir = root_dir
        self.mode = mode
        self.num_points = int(num_points)

        folder = "train_data" if mode == "train" else "test_data"
        self.kinect_dir = os.path.join(root_dir, "data", folder, "kinect")
        self.lidar_dir  = os.path.join(root_dir, "data", folder, "lidar")

        self.kinect_sor = bool(kinect_sor)
        self.lidar_voxel = float(lidar_voxel)
        self.lidar_sor = bool(lidar_sor)

        self.src_sampling = src_sampling
        self.tgt_sampling = tgt_sampling

        self.sep_min_factor = float(sep_min_factor)
        self.sep_max_factor = float(sep_max_factor)
        self.src_jitter_ratio = float(src_jitter_ratio)
        self.max_angle = np.deg2rad(float(max_rot_deg))

        self.max_cache_points = int(max_cache_points)
        self.preload_cache = bool(preload_cache)

        k_files = _find_files(self.kinect_dir)
        l_files = _find_files(self.lidar_dir)
        common = sorted(set(k_files.keys()).intersection(set(l_files.keys())))
        self.pairs = [(k_files[name], l_files[name], name) for name in common]

        print(f"[PAIRED DATA] {mode.upper()}")
        print(f"  Kinect dir: {self.kinect_dir} ({len(k_files)} fichiers)")
        print(f"  LiDAR  dir: {self.lidar_dir} ({len(l_files)} fichiers)")
        print(f"  Pairs found: {len(self.pairs)}")

        self.k_cache = {}  # name -> pts (centered + preprocessed)
        self.l_cache = {}  # name -> pts (centered + preprocessed)

        if self.preload_cache:
            self._preload_all()

    def _preload_all(self):
        for (k_path, l_path, name) in self.pairs:
            _ = self._get_kinect_cached(k_path, name)
            _ = self._get_lidar_cached(l_path, name)

    def _get_kinect_cached(self, k_path: str, name: str) -> np.ndarray:
        if name in self.k_cache:
            return self.k_cache[name]

        pts = _load_points(k_path)
        if pts.shape[0] > self.max_cache_points:
            idx = np.random.choice(pts.shape[0], self.max_cache_points, replace=False)
            pts = pts[idx]

        if self.kinect_sor:
            pts = _statistical_outlier_removal_np(pts, nb_neighbors=20, std_ratio=2.0)

        pts, _ = _center_only(pts)
        self.k_cache[name] = pts
        return pts

    def _get_lidar_cached(self, l_path: str, name: str) -> np.ndarray:
        if name in self.l_cache:
            return self.l_cache[name]

        pts = _load_points(l_path)
        if pts.shape[0] > self.max_cache_points:
            idx = np.random.choice(pts.shape[0], self.max_cache_points, replace=False)
            pts = pts[idx]

        if self.lidar_voxel > 0:
            pts = _voxel_downsample_np(pts, self.lidar_voxel)

        if self.lidar_sor:
            pts = _statistical_outlier_removal_np(pts, nb_neighbors=20, std_ratio=2.0)

        pts, _ = _center_only(pts)
        self.l_cache[name] = pts
        return pts

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        k_path, l_path, name = self.pairs[idx]

        k_pts = self._get_kinect_cached(k_path, name)
        l_pts = self._get_lidar_cached(l_path, name)

        N = self.num_points
        src = _random_sample(k_pts, N) if self.src_sampling == "random" else _farthest_point_sample(k_pts, N)
        tgt = _farthest_point_sample(l_pts, N) if self.tgt_sampling == "fps" else _random_sample(l_pts, N)

        # random RT + separation
        diag = _aabb_diag(tgt)
        R = _random_rotation(self.max_angle)

        direction = np.random.normal(size=3).astype(np.float32)
        direction /= (np.linalg.norm(direction) + 1e-9)
        dist = float(np.random.uniform(self.sep_min_factor, self.sep_max_factor) * diag)
        t = direction * dist

        src = apply_transform(src, R, t)

        sigma = float(self.src_jitter_ratio * diag)
        if sigma > 0:
            src = src + np.random.normal(0, sigma, size=src.shape).astype(np.float32)

        src_tensor = torch.from_numpy(src).float().transpose(1, 0)
        tgt_tensor = torch.from_numpy(tgt).float().transpose(1, 0)

        R_gt = torch.from_numpy(R.T.copy()).float()
        t_gt = torch.from_numpy(-(R.T @ t)).float()

        return src_tensor, tgt_tensor, R_gt, t_gt, name
