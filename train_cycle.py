import os
import sys
import time
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# --- 1. GESTION ROBUSTE DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "data"))
sys.path.append(os.path.join(BASE_DIR, "utils"))

# --- 2. IMPORTS SÉCURISÉS ---
try:
    from rpmnet import RPMNet
except ImportError:
    from models.rpmnet import RPMNet

try:
    from dataset import CrossSourceDataset
except ImportError:
    from data.dataset import CrossSourceDataset

try:
    from transform import transform_point_cloud_torch
except ImportError:
    from utils.transform import transform_point_cloud_torch

from matrics import compute_rmse_cloudcompare, compute_lcp_sklearn

# --- 3. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 4       # Petit batch pour éviter OOM (Out Of Memory) avec le cycle
LR = 1e-4
NUM_POINTS = 512
NUM_ITERS = 5

# Poids des Loss
W_GEOM = 1.0         # Alignement géométrique
W_CYCLE = 0.5        # Cohérence du cycle (Aller-Retour)

# Dossiers de sortie
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_cycle_log.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_geometric_loss(src, tgt, transforms, loss_fn):
    """
    Calcule la loss sur toutes les itérations de RPMNet.
    """
    total_loss = 0.0
    n_iters = len(transforms)
    for i, (r_iter, t_iter) in enumerate(transforms):
        # Poids croissant : on veut surtout que la dernière itération soit bonne
        w = 1.0 / (2 ** (n_iters - 1 - i))
        
        src_transformed = transform_point_cloud_torch(src, r_iter, t_iter)
        total_loss += w * loss_fn(src_transformed, tgt)
    return total_loss


def compute_cycle_loss(R_ab, t_ab, R_ba, t_ba):
    """
    Vérifie que T_ba * T_ab ≈ Identité
    """
    # R_cycle = R_ba @ R_ab
    R_cycle = torch.matmul(R_ba, R_ab)
    
    # t_cycle = R_ba @ t_ab + t_ba
    t_cycle = torch.matmul(R_ba, t_ab.unsqueeze(2)).squeeze(2) + t_ba

    B = R_ab.shape[0]
    I = torch.eye(3, device=R_ab.device).unsqueeze(0).repeat(B, 1, 1)
    zeros = torch.zeros_like(t_cycle)

    loss_r = nn.MSELoss()(R_cycle, I)
    loss_t = nn.MSELoss()(t_cycle, zeros)
    
    return loss_r + loss_t


def train():
    print(f"=== DÉMARRAGE SUR {DEVICE} ===")

    # 1. Chargement du Dataset
    print(f"[INIT] Recherche des données dans {BASE_DIR}...")
    try:
        train_dataset = CrossSourceDataset(BASE_DIR, mode='train', num_points=NUM_POINTS)
        test_dataset  = CrossSourceDataset(BASE_DIR, mode='test',  num_points=NUM_POINTS)
    except Exception as e:
        print(f"[ERREUR CRITIQUE] Impossible de charger le dataset : {e}")
        return

    print(f"[DATA] Train: {len(train_dataset)} paires | Test: {len(test_dataset)} paires")

    if len(train_dataset) == 0:
        print("[STOP] Le dataset TRAIN est vide. Vérifie tes dossiers 'data/train_data/kinect' et 'lidar'.")
        print("       Les fichiers doivent avoir EXACTEMENT le même nom dans les deux dossiers.")
        return

    # Drop_last=False pour éviter de crasher si le dataset est tout petit (< batch_size)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # 2. Modèle
    print("[INIT] Chargement du modèle RPMNet...")
    model = RPMNet(num_iterations=NUM_ITERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    mse_loss = nn.MSELoss()

    # Création du header CSV
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Total_Loss", "Geom_Loss", "Cycle_Loss", "Val_RMSE", "Val_LCP"])

    print("=== DÉBUT DE L'ENTRAÎNEMENT ===")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_geom = 0.0
        epoch_cycle = 0.0
        
        start_time = time.time()
        batch_count = 0

        for i, (src, tgt, _, _, _) in enumerate(train_loader):
            batch_count += 1
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            optimizer.zero_grad()
            
            # --- A. SENS 1 : Kinect -> Lidar ---
            # src=Kinect (déformé), tgt=Lidar (fixe)
            R_kl, t_kl, trans_kl = model(src, tgt)
            loss_geom_kl = compute_geometric_loss(src, tgt, trans_kl, mse_loss)
            
            # --- B. SENS 2 : Lidar -> Kinect ---
            # On inverse : src=Lidar, tgt=Kinect
            # Note : Pour la Cycle Loss pure, on réutilise les tenseurs du batch.
            R_lk, t_lk, trans_lk = model(tgt, src) 
            loss_geom_lk = compute_geometric_loss(tgt, src, trans_lk, mse_loss)

            # --- C. CYCLE CONSISTENCY ---
            loss_cycle = compute_cycle_loss(R_kl, t_kl, R_lk, t_lk)

            # TOTAL
            loss = W_GEOM * (loss_geom_kl + loss_geom_lk) + W_CYCLE * loss_cycle
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_geom += (loss_geom_kl.item() + loss_geom_lk.item())
            epoch_cycle += loss_cycle.item()

        # Sécurité si le loader ne tourne pas
        if batch_count == 0:
            print("[ERREUR] Aucune batch n'a été traitée. Vérifie le dataset.")
            break

        # Moyennes
        avg_loss = epoch_loss / batch_count
        avg_geom = epoch_geom / batch_count
        avg_cycle = epoch_cycle / batch_count

        # --- VALIDATION (Forward seulement) ---
        model.eval()
        val_rmse_list = []
        val_lcp_list = []
        
        with torch.no_grad():
            for src, tgt, _, _, _ in test_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                R_pred, t_pred, _ = model(src, tgt)
                src_final = transform_point_cloud_torch(src, R_pred, t_pred)
                
                # Metrics CPU
                p_pred = src_final.transpose(1, 2).cpu().numpy() # (B, N, 3)
                p_tgt = tgt.transpose(1, 2).cpu().numpy()        # (B, N, 3)
                
                for b in range(src.size(0)):
                    val_rmse_list.append(compute_rmse_cloudcompare(p_pred[b], p_tgt[b]))
                    val_lcp_list.append(compute_lcp_sklearn(p_pred[b], p_tgt[b]))

        val_rmse = np.mean(val_rmse_list) if val_rmse_list else 0
        val_lcp = np.mean(val_lcp_list) if val_lcp_list else 0
        
        scheduler.step(avg_loss)

        # Log Console
        duration = time.time() - start_time
        print(f"Ep {epoch:03d} | Loss {avg_loss:.4f} (G:{avg_geom:.2f} C:{avg_cycle:.2f}) | Val RMSE: {val_rmse:.4f} LCP: {val_lcp:.2f} | {duration:.1f}s")
        
        # Log CSV
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss, avg_geom, avg_cycle, val_rmse, val_lcp])

        # Save
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"rpm_cycle_ep{epoch}.pth"))

# --- POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    # Cette condition est obligatoire sous Windows pour le multiprocessing (si num_workers > 0)
    # Et elle assure que le code se lance bien.
    train()