import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d
import time
import copy # Pour la fusion propre des nuages

from data.dataset import SingleSourceDataset
from models.rpmnet import RPMNet
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

EPOCHS = 30       
BATCH_SIZE = 8    
LR = 0.00005

EPOCHS = 30       # Pas besoin de plus si Ep5 est déjà bon
BATCH_SIZE = 8    
LR = 0.00005      # <--- J'ai baissé ça (c'était 0.0001). Plus doux pour ne pas "casser" la perfection.


def save_combined_ply(pcd1_tensor, color1, pcd2_tensor, color2, filename):
    """ Fusionne physiquement deux nuages dans un seul fichier .ply """
    
    # Conversion Tensor -> Numpy
    pts1 = pcd1_tensor.cpu().detach().numpy().T
    pts2 = pcd2_tensor.cpu().detach().numpy().T
    
    # Création des objets Open3D
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(pts1)

    cloud1.paint_uniform_color(color1)
    
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(pts2)
    cloud2.paint_uniform_color(color2) 
    
    # Fusion

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    mse_loss = nn.MSELoss()

    model.train()

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


        # SAUVEGARDE À CHAQUE EPOCH
        # 1. Le Cerveau
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"rpm_epoch_{epoch+1}.pth"))
        
        # 2. Les Images

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
             

             # SAUVEGARDE RESULTAT
             save_combined_ply(
                 res_0, [1, 0, 0],  # Bleu
             )
             # SAUVEGARDE RESULTAT (Bleu + Vert)
             save_combined_ply(
                 res_0, [0, 0, 1],  # Bleu

                 tgt_0, [0, 1, 0],  # Vert
                 os.path.join(VISUAL_DIR, f"Ep{epoch+1}_FINAL.ply")
             )
             
             print(f"   -> Ep{epoch+1} sauvegardée.")

if __name__ == "__main__":
    train()