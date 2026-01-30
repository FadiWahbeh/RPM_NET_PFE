"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d
import time
import copy # Pour la fusion propre des nuages
import numpy as np
import tens

from data.dataset import SingleSourceDataset
from models.rpmnet import RPMNet
from models.rmse import rmse
from utils.transform import transform_point_cloud_torch



# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visuals")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REGLAGES FINS ---
EPOCHS = 30       # Pas besoin de plus si Ep5 est déjà bon
BATCH_SIZE = 8    
LR = 0.00005      # <--- J'ai baissé ça (c'était 0.0001). Plus doux pour ne pas "casser" la perfection.

def save_combined_ply(pcd1_tensor, color1, pcd2_tensor, color2, filename):
     #Fusionne physiquement deux nuages dans un seul fichier .ply
    
    # Conversion Tensor -> Numpy
    pts1 = pcd1_tensor.cpu().detach().numpy().T
    pts2 = pcd2_tensor.cpu().detach().numpy().T
    
    # Création des objets Open3D
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(pts1)
    cloud1.paint_uniform_color(color1) # ex: Rouge
    
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(pts2)
    cloud2.paint_uniform_color(color2) # ex: Vert
    
    # Fusion (Concaténation)
    combined = cloud1 + cloud2
    
    # Sauvegarde
    o3d.io.write_point_cloud(filename, combined)

def train():
    print(f"--- TRAIN RPM-NET (Save All Epochs) ---")
    
    dataset = SingleSourceDataset(BASE_DIR, num_points=512)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = RPMNet(num_iterations=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    mse_loss = nn.MSELoss()

    model.train()
    rmse_val_min = 3.4
    eborch_best = 1

    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for src, tgt, true_R, true_t, _ in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            true_R, true_t = true_R.to(DEVICE), true_t.to(DEVICE)
            
            optimizer.zero_grad()
            
            R_pred, t_pred, transforms = model(src, tgt)
            
            loss = 0
            n_iters = len(transforms)
            for i, (r_iter, t_iter) in enumerate(transforms):
                w = 1.0 / (2 ** (n_iters - 1 - i))
                l_r = mse_loss(torch.matmul(r_iter.transpose(1, 2), true_R), torch.eye(3).to(DEVICE).unsqueeze(0).repeat(src.size(0),1,1))
                l_t = mse_loss(t_iter, true_t)
                loss += w * (l_r + l_t)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Time: {time.time() - start_time:.1f}s")

        # --- SAUVEGARDE À CHAQUE EPOCH (1, 2, 3...) ---
        # Plus de modulo %, on sauvegarde tout.
        
        # 1. Le Cerveau (.pth)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"rpm_epoch_{epoch+1}.pth"))
        
        # 2. Les Images (.ply)
        with torch.no_grad():
            
             # On prend le premier exemple du batch
             src_0 = src[0]
             tgt_0 = tgt[0]
             
             # Résultat IA
             res_0 = transform_point_cloud_torch(src[0:1], R_pred[0:1], t_pred[0:1])[0]

             # SAUVEGARDE DEPART (Rouge + Vert)
             save_combined_ply(
                 src_0, [1, 0, 0],  # Rouge
                 tgt_0, [0, 1, 0],  # Vert
                 os.path.join(VISUAL_DIR, f"Ep{epoch+1}_DEPART.ply")
             )
             
             # SAUVEGARDE RESULTAT (Bleu + Vert)
             save_combined_ply(
                 res_0, [0, 0, 1],  # Bleu
                 tgt_0, [0, 1, 0],  # Vert
                 os.path.join(VISUAL_DIR, f"Ep{epoch+1}_FINAL.ply")
             )
             
             print(f"   -> Ep{epoch+1} sauvegardée.")
        
        res = transform_point_cloud_torch(src, R_pred, t_pred)
        # Calcul RMSE pour suivi
        rmse_epoch = rmse(src.cpu().detach(), tgt.cpu().detach())
        if rmse_epoch < rmse_val_min:
            rmse_val_min = rmse_epoch
            eborch_best = epoch + 1
        print (f"   -> RMSE Epoch {epoch+1}: {rmse_epoch:.6f}")
        print(f"   -> Nouveau RMSE min: {rmse_val_min:.6f} a l'Epoch {eborch_best}")

if __name__ == "__main__":
    train()
    
    """
    
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d
import time
import numpy as np

from data.dataset import CrossSourceDataset
from models.rpmnet import RPMNet
from matrics import compute_lcp_sklearn, compute_rmse_cloudcompare 
from utils.transform import transform_point_cloud_torch

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visuals")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REGLAGES ---
EPOCHS = 30       
BATCH_SIZE = 8    
LR = 0.00005      

def train():
    print(f"--- TRAIN RPM-NET (Fix: Inverse Ground Truth) ---")
    
    train_dataset = CrossSourceDataset(BASE_DIR, mode='train', num_points=512)
    test_dataset = CrossSourceDataset(BASE_DIR, mode='test', num_points=512)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = RPMNet(num_iterations=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    # MSE Loss standard
    mse_loss = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        
        for src, tgt, true_R, true_t, _ in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            true_R, true_t = true_R.to(DEVICE), true_t.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Le réseau prédit la transfo pour aller de SRC -> TGT
            R_pred, t_pred, transforms = model(src, tgt)
            
            # --- CORRECTION CRUCIALE ---
            # true_R/t est la transfo TGT -> SRC (celle du dataset).
            # Le réseau prédit l'inverse (SRC -> TGT).
            # On doit donc calculer l'inverse de la vérité pour comparer des pommes avec des pommes.
            
            # Inverse Rotation : R^-1 = R^T
            gt_R_inv = true_R.transpose(1, 2)
            
            # Inverse Translation : t^-1 = -R^T * t
            # Attention aux dimensions pour le matmul (Batch, 3, 1)
            gt_t_inv = -torch.matmul(gt_R_inv, true_t.unsqueeze(2)).squeeze(2)
            
            loss = 0
            n_iters = len(transforms)
            for i, (r_iter, t_iter) in enumerate(transforms):
                w = 1.0 / (2 ** (n_iters - 1 - i))
                
                # On compare la prédiction avec l'INVERSE de la vérité
                # Astuce math: Si r_iter est bon, r_iter * gt_R doit donner l'Identité.
                # Ou plus simple: MSE(r_iter, gt_R_inv)
                
                l_r = mse_loss(r_iter, gt_R_inv) 
                l_t = mse_loss(t_iter, gt_t_inv)
                
                loss += w * (l_r + l_t)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_lcp_list = []
        val_rmse_list = []
        last_batch_data = None 
        
        with torch.no_grad():
            for src, tgt, true_R, true_t, fnames in test_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                true_R, true_t = true_R.to(DEVICE), true_t.to(DEVICE)
                
                R_pred, t_pred, _ = model(src, tgt)
                src_transformed = transform_point_cloud_torch(src, R_pred, t_pred)
                
                for i in range(src.size(0)):
                    p_pred = src_transformed[i].transpose(0, 1)
                    p_tgt = tgt[i].transpose(0, 1)
                    val_lcp_list.append(compute_lcp_sklearn(p_pred, p_tgt))
                    val_rmse_list.append(compute_rmse_cloudcompare(p_pred, p_tgt))
                
                last_batch_data = (src, tgt, R_pred, t_pred, true_R, true_t, fnames)

        avg_lcp = np.mean(val_lcp_list)
        avg_rmse = np.mean(val_rmse_list)
        scheduler.step(avg_train_loss)
        
        print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f} | Val LCP: {avg_lcp:.2%} | Val RMSE: {avg_rmse:.4f}")

        # --- SAUVEGARDE ---
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"rpm_epoch_{epoch+1}.pth"))
        
        if last_batch_data is not None:
             src_b, tgt_b, R_pred_b, t_pred_b, true_R_b, true_t_b, fnames_b = last_batch_data
             
             R_pred_np = R_pred_b[0].cpu().detach().numpy()
             t_pred_np = t_pred_b[0].cpu().detach().numpy()
             R_true_np = true_R_b[0].cpu().detach().numpy()
             t_true_np = true_t_b[0].cpu().detach().numpy()
             
             original_path = os.path.join(test_dataset.src_dir, fnames_b[0])
             
             try:
                 pcd_hd = o3d.io.read_point_cloud(original_path)
                 pts_hd_base = np.asarray(pcd_hd.points)

                 if len(pts_hd_base) > 10000:
                     indices = np.random.choice(len(pts_hd_base), 10000, replace=False)
                     pts_hd_base = pts_hd_base[indices]
                 
                 pts_hd_base = pts_hd_base - np.mean(pts_hd_base, axis=0)
                 max_d = np.max(np.sqrt(np.sum(pts_hd_base**2, axis=1)))
                 if max_d > 0: pts_hd_base /= max_d
                 
                 # 1. Target (Vert)
                 cloud_tgt = o3d.geometry.PointCloud()
                 cloud_tgt.points = o3d.utility.Vector3dVector(pts_hd_base)
                 cloud_tgt.paint_uniform_color([0, 1, 0]) 

                 # 2. Source Init (Rouge)
                 # On recrée l'état initial avec la transformation "Dataset" (R_true)
                 pts_src_init = pts_hd_base @ R_true_np.T + t_true_np
                 cloud_src_init = o3d.geometry.PointCloud()
                 cloud_src_init.points = o3d.utility.Vector3dVector(pts_src_init)
                 cloud_src_init.paint_uniform_color([1, 0, 0]) 

                 # 3. Résultat (Bleu)
                 # On applique la correction du réseau (R_pred) sur la Source Initiale
                 pts_src_final = pts_src_init @ R_pred_np.T + t_pred_np
                 cloud_src_final = o3d.geometry.PointCloud()
                 cloud_src_final.points = o3d.utility.Vector3dVector(pts_src_final)
                 cloud_src_final.paint_uniform_color([0, 0, 1])
                 
                 o3d.io.write_point_cloud(os.path.join(VISUAL_DIR, f"Ep{epoch+1}_Target_Ref.ply"), cloud_tgt)
                 o3d.io.write_point_cloud(os.path.join(VISUAL_DIR, f"Ep{epoch+1}_Start_Rouge.ply"), cloud_src_init)
                 o3d.io.write_point_cloud(os.path.join(VISUAL_DIR, f"Ep{epoch+1}_Result_Bleu.ply"), cloud_src_final)

                 print(f"   -> Fichiers sauvegardés (Target, Start, Result).")
                 
             except Exception as e: print(f"   -> Erreur sauvegarde: {e}")

if __name__ == "__main__":
    train()