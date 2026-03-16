import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import glob

SUPPORTED_EXTS = ('.ply', '.pcd', '.xyz', '.txt', '.pts', '.npy', '.npz')

# FPS pour sous-échantillonnage.
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

# Trouve les paires à partir de la structure de dossiers.
def find_pairs_from_folders(data_dir, source_sensor='lidar', target_sensor='kinect'):
    source_dir = os.path.join(data_dir, source_sensor)
    target_dir = os.path.join(data_dir, target_sensor)
    
    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        print(f"Attention: Dossiers non trouvés: {source_dir} ou {target_dir}")
        return []
    
    source_files = []
    for ext in SUPPORTED_EXTS:
        source_files.extend(glob.glob(os.path.join(source_dir, f'*{ext}')))
        source_files.extend(glob.glob(os.path.join(source_dir, f'*{ext.upper()}')))
    
    pairs = []
    for src_path in source_files:
        src_basename = os.path.basename(src_path)
        src_name = os.path.splitext(src_basename)[0]
        
        tgt_path = None
        for ext in SUPPORTED_EXTS:
            candidate = os.path.join(target_dir, src_basename)
            if os.path.exists(candidate):
                tgt_path = candidate
                break
            
            candidate = os.path.join(target_dir, src_name + ext)
            if os.path.exists(candidate):
                tgt_path = candidate
                break
        
        if tgt_path is not None:
            pairs.append({'source': src_path, 'target': tgt_path, 'scene': src_name})
        else:
            print(f"Attention: Pas de correspondant trouvé pour {src_basename} dans {target_dir}")
    
    return pairs

# Estime les normales pour un nuage de points.
def estimate_normals(points, k=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals).astype(np.float32)

# Calcule les descripteurs FPFH.
def compute_fpfh(points, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )
    return np.asarray(fpfh.data).T  # (N, 33)


class CrossSourceDataset(Dataset):
    def __init__(self, root_dir, mode='train', num_points=1024,
                 source_sensor='lidar', target_sensor='kinect',
                 use_normals=True, use_fpfh=False):
        
        if mode == 'train':
            self.data_dir = os.path.join(root_dir, 'data', 'train_data')
        else:
            self.data_dir = os.path.join(root_dir, 'data', 'test_data')
        
        self.num_points = int(num_points)
        self.source_sensor = source_sensor
        self.target_sensor = target_sensor
        self.use_normals = use_normals
        self.use_fpfh = use_fpfh
        
        source_full_path = os.path.join(self.data_dir, source_sensor)
        target_full_path = os.path.join(self.data_dir, target_sensor)
        
        print(f"\n[DATA] Recherche dans:")
        print(f"  Source: {source_full_path}")
        print(f"  Target: {target_full_path}")
        
        if not os.path.exists(source_full_path):
            print(f"  [WARN] Dossier source n'existe pas: {source_full_path}")
        if not os.path.exists(target_full_path):
            print(f"  [WARN] Dossier target n'existe pas: {target_full_path}")
        
        self.pairs = find_pairs_from_folders(self.data_dir, source_sensor=source_sensor, target_sensor=target_sensor)
        
        print(f"\n[DATA] Chargement {mode.upper()} depuis {self.data_dir}")
        print(f"      {len(self.pairs)} paires cross-source trouvées")
        print(f"      Source: {source_sensor} -> Target: {target_sensor}")
        print(f"      Normales: {use_normals}, FPFH: {use_fpfh}")
        
        self.cache = {}
        
        if len(self.pairs) == 0:
            print("\n[ERREUR] Aucune paire cross-source trouvée.")
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_points(self, path):
        if path in self.cache:
            return self.cache[path]['points'], self.cache[path]['raw']
        
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.ply', '.pcd'):
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points).astype(np.float32)
        elif ext in ('.xyz', '.txt', '.pts'):
            pts = np.loadtxt(path, dtype=np.float32)
            if pts.ndim == 1: pts = pts[None, :]
            pts = pts[:, :3].astype(np.float32)
        elif ext == '.npy':
            pts = np.load(path).astype(np.float32)[:, :3]
        elif ext == '.npz':
            data = np.load(path)
            for k in ['points', 'xyz', 'pc', 'arr_0']:
                if k in data:
                    pts = data[k].astype(np.float32)[:, :3]
                    break
        
        self.cache[path] = {'points': pts, 'raw': pts.copy()}
        return pts, pts
    
    def _preprocess(self, points, compute_normals=True):
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        result = {'points': points_centered, 'mean': mean, 'raw': points}
        
        if compute_normals and (self.use_normals or self.use_fpfh):
            try:
                result['normals'] = estimate_normals(points)
            except Exception as e:
                print(f"Warning: Impossible de calculer les normales: {e}")
                result['normals'] = np.zeros_like(points)
        return result
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        try:
            src_points, _ = self._load_points(pair['source'])
            tgt_points, _ = self._load_points(pair['target'])
            
            src_data = self._preprocess(src_points)
            tgt_data = self._preprocess(tgt_points)
            
            src_input = src_data['points']
            tgt_input = tgt_data['points']
            
            if self.use_normals and 'normals' in src_data:
                src_input = np.concatenate([src_input, src_data['normals']], axis=1)
            if self.use_normals and 'normals' in tgt_data:
                tgt_input = np.concatenate([tgt_input, tgt_data['normals']], axis=1)
            
            # Le FPS se fera uniquement sur le XYZ
            src_sampled = farthest_point_sample(src_input, self.num_points)
            tgt_sampled = farthest_point_sample(tgt_input, self.num_points)
            
            src_pts = src_sampled[:, :3]
            tgt_pts = tgt_sampled[:, :3]
            
            # src_sampled contient désormais (N, 6) si use_normals=True
            src_features = [src_sampled]
            tgt_features = [tgt_sampled]
            
            if self.use_fpfh and 'normals' in src_data and 'normals' in tgt_data:
                try:
                    src_n = src_sampled[:, 3:6] if self.use_normals else estimate_normals(src_pts)
                    tgt_n = tgt_sampled[:, 3:6] if self.use_normals else estimate_normals(tgt_pts)
                    
                    src_features.append(compute_fpfh(src_pts, src_n))
                    tgt_features.append(compute_fpfh(tgt_pts, tgt_n))
                except Exception as e:
                    pass
            
            src_enhanced = np.concatenate(src_features, axis=1).astype(np.float32)
            tgt_enhanced = np.concatenate(tgt_features, axis=1).astype(np.float32)
            
            return {
                'src': torch.from_numpy(src_enhanced).float(),      
                'tgt': torch.from_numpy(tgt_enhanced).float(),      
                'src_xyz': torch.from_numpy(src_pts).float(),       
                'tgt_xyz': torch.from_numpy(tgt_pts).float(),       
                'R': torch.eye(3).float(),
                't': torch.zeros(3).float(),
                'scene': pair['scene'],
                'src_path': pair['source'],
                'tgt_path': pair['target']
            }
            
        except Exception as e:
            dummy = torch.zeros((self.num_points, 6 if self.use_normals else 3))
            dummy_xyz = torch.zeros((self.num_points, 3))
            return {
                'src': dummy, 'tgt': dummy, 'src_xyz': dummy_xyz, 'tgt_xyz': dummy_xyz,
                'R': torch.eye(3), 't': torch.zeros(3), 'scene': 'error',
                'src_path': '', 'tgt_path': ''
            }