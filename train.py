import os
import sys
import time
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d

# =========================
# SETTINGS
# =========================
RESUME_TRAINING = False
SAVE_BEST = True
BEST_NAME = "rpm_best.pth"

# LCP adaptatif : threshold = ratio * diag
LCP_RATIO = 0.02  # 2% de la taille (0.01 ou 0.05 si besoin)

# RMSE unité + RMSE normalisée (% de la taille)
RMSE_UNIT = "m"          # mets "mm" si tes coordonnées sont en millimètres
LOG_RMSE_NORM = True     # ajoute RMSE (%) dans logs/print

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UTILS_DIR = os.path.join(BASE_DIR, "utils")
for p in [BASE_DIR, DATA_DIR, MODELS_DIR, UTILS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from matrics import compute_rmse_cloudcompare, compute_lcp_adaptive

try:
    from data.dataset import CrossSourceDataset
except Exception:
    from dataset import CrossSourceDataset

try:
    from models.rpmnet import RPMNet
except Exception:
    from rpmnet import RPMNet

try:
    from utils.transform import transform_point_cloud_torch
except Exception:
    from transform import transform_point_cloud_torch


# OUTPUTS
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIS_TRAIN_DIR = os.path.join(OUTPUT_DIR, "visuals_train")
VIS_TEST_DIR  = os.path.join(OUTPUT_DIR, "visuals_test")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "Plot")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")

os.makedirs(VIS_TRAIN_DIR, exist_ok=True)
os.makedirs(VIS_TEST_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMS
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-2
NUM_POINTS = 512
NUM_ITERS = 5

SAVE_EVERY = 1
HD_MAX_POINTS = 200_000
UHD_MAX_POINTS = 2_000_000
EXPORT_UHD = False

# Helpers

def batch_diag_from_tgt(tgt_b3n: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """diag(AABB) par batch. tgt: (B,3,N) -> (B,)"""
    pmin = tgt_b3n.amin(dim=2)  # (B,3)
    pmax = tgt_b3n.amax(dim=2)  # (B,3)
    diag = torch.norm(pmax - pmin, dim=1)  # (B,)
    return torch.clamp(diag, min=eps)

def rmse_percent_of_size(rmse_abs: float, diag: float) -> float:
    """RMSE normalisée en pourcentage de la taille (diag AABB)."""
    if diag <= 1e-12:
        return 0.0
    return 100.0 * (rmse_abs / diag)

def diag_np_from_points(points_np: np.ndarray) -> float:
    """points_np: (N,3)"""
    pmin = points_np.min(axis=0)
    pmax = points_np.max(axis=0)
    return float(np.linalg.norm(pmax - pmin))

def save_ply(points_np, colors_np, out_path):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_np.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float32))
    o3d.io.write_point_cloud(out_path, cloud)

def save_colored_single(points_np, rgb, out_path):
    colors = np.tile(np.array(rgb, dtype=np.float32)[None, :], (points_np.shape[0], 1))
    save_ply(points_np, colors, out_path)

def save_colored_pair(pointsA, rgbA, pointsB, rgbB, out_path):
    A = pointsA.astype(np.float32)
    B = pointsB.astype(np.float32)
    P = np.vstack([A, B])
    CA = np.tile(np.array(rgbA, dtype=np.float32)[None, :], (A.shape[0], 1))
    CB = np.tile(np.array(rgbB, dtype=np.float32)[None, :], (B.shape[0], 1))
    C = np.vstack([CA, CB])
    save_ply(P, C, out_path)

def load_full_cloud_centered(dataset_obj, fname, max_points=None):
    path = dataset_obj.get_file_path(fname)
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float32)

    if max_points is not None and len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    mean, _scale = dataset_obj.get_norm_params(fname)
    pts = (pts - mean).astype(np.float32)
    return pts

def apply_rt_np(points, R, t):
    return points @ R.T + t

def export_step_by_step(epoch_idx, base_dir,
                        src_tensor, tgt_tensor, transforms,
                        fname0, dataset_obj,
                        true_R, true_t):
    ep_dir = os.path.join(base_dir, f"Ep{epoch_idx:03d}")
    low_dir = os.path.join(ep_dir, "low")
    hd_dir  = os.path.join(ep_dir, "hd")
    uhd_dir = os.path.join(ep_dir, "ultrahd")

    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(hd_dir, exist_ok=True)
    if EXPORT_UHD:
        os.makedirs(uhd_dir, exist_ok=True)

    # LOW (b=0)
    src0 = src_tensor[0].detach().cpu().numpy().T
    tgt0 = tgt_tensor[0].detach().cpu().numpy().T

    save_colored_single(tgt0, [0, 1, 0], os.path.join(low_dir, "Target_low.ply"))
    save_colored_single(src0, [1, 0, 0], os.path.join(low_dir, "Start_low.ply"))
    save_colored_pair(src0, [1, 0, 0], tgt0, [0, 1, 0], os.path.join(low_dir, "Start_plus_Target_low.ply"))

    for i, (r_it, t_it) in enumerate(transforms):
        src_it = transform_point_cloud_torch(src_tensor, r_it, t_it)[0].detach().cpu().numpy().T
        save_colored_single(src_it, [0, 0, 1], os.path.join(low_dir, f"Result_it{i+1}_low.ply"))
        save_colored_pair(src_it, [0, 0, 1], tgt0, [0, 1, 0], os.path.join(low_dir, f"Result_it{i+1}_plus_Target_low.ply"))

    # HD
    tgt_hd = load_full_cloud_centered(dataset_obj, fname0, max_points=HD_MAX_POINTS)
    R_true = true_R[0].detach().cpu().numpy()
    t_true = true_t[0].detach().cpu().numpy()
    start_hd = apply_rt_np(tgt_hd, R_true, t_true)

    save_colored_single(tgt_hd, [0, 1, 0], os.path.join(hd_dir, "Target_HD.ply"))
    save_colored_single(start_hd, [1, 0, 0], os.path.join(hd_dir, "Start_HD.ply"))
    save_colored_pair(start_hd, [1, 0, 0], tgt_hd, [0, 1, 0], os.path.join(hd_dir, "Start_plus_Target_HD.ply"))

    for i, (r_it, t_it) in enumerate(transforms):
        R_it = r_it[0].detach().cpu().numpy()
        t_it_np = t_it[0].detach().cpu().numpy()
        res_hd = apply_rt_np(start_hd, R_it, t_it_np)
        save_colored_single(res_hd, [0, 0, 1], os.path.join(hd_dir, f"Result_it{i+1}_HD.ply"))
        save_colored_pair(res_hd, [0, 0, 1], tgt_hd, [0, 1, 0], os.path.join(hd_dir, f"Result_it{i+1}_plus_Target_HD.ply"))

    if EXPORT_UHD:
        tgt_uhd = load_full_cloud_centered(dataset_obj, fname0, max_points=UHD_MAX_POINTS)
        start_uhd = apply_rt_np(tgt_uhd, R_true, t_true)

        save_colored_single(tgt_uhd, [0, 1, 0], os.path.join(uhd_dir, "Target_UHD.ply"))
        save_colored_single(start_uhd, [1, 0, 0], os.path.join(uhd_dir, "Start_UHD.ply"))
        save_colored_pair(start_uhd, [1, 0, 0], tgt_uhd, [0, 1, 0], os.path.join(uhd_dir, "Start_plus_Target_UHD.ply"))

        for i, (r_it, t_it) in enumerate(transforms):
            R_it = r_it[0].detach().cpu().numpy()
            t_it_np = t_it[0].detach().cpu().numpy()
            res_uhd = apply_rt_np(start_uhd, R_it, t_it_np)
            save_colored_single(res_uhd, [0, 0, 1], os.path.join(uhd_dir, f"Result_it{i+1}_UHD.ply"))
            save_colored_pair(res_uhd, [0, 0, 1], tgt_uhd, [0, 1, 0], os.path.join(uhd_dir, f"Result_it{i+1}_plus_Target_UHD.ply"))


# TRAIN

def train():
    print(" TRAIN RPM-NET (NO RESUME + BEST + NO RESIZE + SCALE-INVARIANT LOSS)")

    train_dataset = CrossSourceDataset(BASE_DIR, mode='train', num_points=NUM_POINTS)
    test_dataset  = CrossSourceDataset(BASE_DIR, mode='test',  num_points=NUM_POINTS)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(
            "Dataset vide: mets des fichiers dans data/train_data et data/test_data "
            "(formats: .ply .pcd .xyz .txt .pts .npy .npz)"
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = RPMNet(num_iterations=NUM_ITERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    mse_loss = nn.MSELoss()

    start_epoch = 1
    if RESUME_TRAINING:
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("rpm_epoch_") and f.endswith(".pth")]
        if len(ckpts) > 0:
            try:
                latest_ckpt = max(ckpts, key=lambda x: int(x.split('_')[2].split('.')[0]))
                latest_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
                print(f"[RESUME] Chargement : {latest_path}")
                model.load_state_dict(torch.load(latest_path, map_location=DEVICE))
                start_epoch = int(latest_ckpt.split('_')[2].split('.')[0]) + 1
                print(f"[RESUME] Reprise à l'epoch {start_epoch}")
            except Exception as e:
                print(f"[RESUME] Erreur, restart: {e}")
    else:
        print("[INIT] RESUME désactivé : nouveau training à chaque exécution.")

    # BEST
    best_val = float("inf")
    best_path = os.path.join(CHECKPOINT_DIR, BEST_NAME)

    # LOG CSV
    write_header = not os.path.exists(LOG_FILE)
    header = [
        "Epoch",
        "Train_Loss", "Val_Loss",
        "Train_RMSE", "Val_RMSE",
        "Train_RMSE_Unit", "Val_RMSE_Unit",
        "Train_RMSE_Pct", "Val_RMSE_Pct",
        "Train_LCP",  "Val_LCP",
        "Epoch_Time_s", "Total_Time_s",
        "Train_IterTimeMean_s", "Val_IterTimeMean_s"
    ]
    if write_header:
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(header)

    t_total0 = time.perf_counter()

    for epoch in range(start_epoch, EPOCHS + 1):
        t_epoch0 = time.perf_counter()

        # TRAIN
        model.train()
        total_loss = 0.0
        rmse_list = []
        rmse_pct_list = []
        lcp_list = []
        iter_time_list = []
        export_train_pack = None

        for src, tgt, true_R, true_t, fnames in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            true_R, true_t = true_R.to(DEVICE), true_t.to(DEVICE)

            optimizer.zero_grad()

            R_pred, t_pred, transforms, iter_times = model(src, tgt, return_timings=True)
            iter_time_list.append(float(np.mean(iter_times)))

            # diag par batch (pour normaliser l'erreur)
            diag = batch_diag_from_tgt(tgt)         # (B,)
            diag_b = diag.view(-1, 1, 1)            # (B,1,1)

            # LOSS SCALE-INVARIANT: MSE((src-tgt)/diag)
            loss = 0.0
            n_iters = len(transforms)
            for i, (r_iter, t_iter) in enumerate(transforms):
                w = 1.0 / (2 ** (n_iters - 1 - i))  # plus de poids sur la fin
                src_iter = transform_point_cloud_torch(src, r_iter, t_iter)
                err = (src_iter - tgt) / diag_b
                loss += w * mse_loss(err, torch.zeros_like(err))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())

            # metrics sur final (ABS RMSE + RMSE% + LCP adaptatif)
            src_final = transform_point_cloud_torch(src, R_pred, t_pred)
            for b in range(src.size(0)):
                p_pred = src_final[b].transpose(0, 1)
                p_tgt  = tgt[b].transpose(0, 1)

                rmse_abs = compute_rmse_cloudcompare(p_pred, p_tgt)
                rmse_list.append(rmse_abs)

                if LOG_RMSE_NORM:
                    p_tgt_np = p_tgt.detach().cpu().numpy()
                    diag_np = diag_np_from_points(p_tgt_np)
                    rmse_pct_list.append(rmse_percent_of_size(rmse_abs, diag_np))

                lcp_list.append(compute_lcp_adaptive(p_pred, p_tgt, ratio=LCP_RATIO))

            if export_train_pack is None:
                export_train_pack = (src.detach(), tgt.detach(), transforms, fnames[0], true_R.detach(), true_t.detach())

        train_loss = total_loss / max(1, len(train_loader))
        train_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
        train_rmse_pct = float(np.mean(rmse_pct_list)) if (LOG_RMSE_NORM and rmse_pct_list) else 0.0
        train_lcp  = float(np.mean(lcp_list)) if lcp_list else 0.0
        train_iter_time = float(np.mean(iter_time_list)) if iter_time_list else 0.0

        # VAL
        model.eval()
        val_loss_acc = 0.0
        val_rmse_list = []
        val_rmse_pct_list = []
        val_lcp_list  = []
        val_iter_time_list = []
        export_test_pack = None

        with torch.no_grad():
            for src, tgt, true_R, true_t, fnames in test_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                true_R, true_t = true_R.to(DEVICE), true_t.to(DEVICE)

                R_pred, t_pred, transforms, iter_times = model(src, tgt, return_timings=True)
                val_iter_time_list.append(float(np.mean(iter_times)))

                diag = batch_diag_from_tgt(tgt)
                diag_b = diag.view(-1, 1, 1)

                last_r, last_t = transforms[-1]
                src_final_val = transform_point_cloud_torch(src, last_r, last_t)

                # VAL LOSS SCALE-INVARIANT
                err = (src_final_val - tgt) / diag_b
                val_loss_acc += float(mse_loss(err, torch.zeros_like(err)).item())

                for b in range(src.size(0)):
                    p_pred = src_final_val[b].transpose(0, 1)
                    p_tgt  = tgt[b].transpose(0, 1)

                    rmse_abs = compute_rmse_cloudcompare(p_pred, p_tgt)
                    val_rmse_list.append(rmse_abs)

                    if LOG_RMSE_NORM:
                        p_tgt_np = p_tgt.detach().cpu().numpy()
                        diag_np = diag_np_from_points(p_tgt_np)
                        val_rmse_pct_list.append(rmse_percent_of_size(rmse_abs, diag_np))

                    val_lcp_list.append(compute_lcp_adaptive(p_pred, p_tgt, ratio=LCP_RATIO))

                if export_test_pack is None:
                    export_test_pack = (src.detach(), tgt.detach(), transforms, fnames[0], true_R.detach(), true_t.detach())

        val_loss = val_loss_acc / max(1, len(test_loader))
        val_rmse = float(np.mean(val_rmse_list)) if val_rmse_list else 0.0
        val_rmse_pct = float(np.mean(val_rmse_pct_list)) if (LOG_RMSE_NORM and val_rmse_pct_list) else 0.0
        val_lcp  = float(np.mean(val_lcp_list)) if val_lcp_list else 0.0
        val_iter_time = float(np.mean(val_iter_time_list)) if val_iter_time_list else 0.0

        scheduler.step(val_loss)

        t_epoch1 = time.perf_counter()
        epoch_time = t_epoch1 - t_epoch0
        total_time = t_epoch1 - t_total0

        train_rmse_str = f"{train_rmse:.4f}{RMSE_UNIT}"
        val_rmse_str   = f"{val_rmse:.4f}{RMSE_UNIT}"
        if LOG_RMSE_NORM:
            train_rmse_str += f" ({train_rmse_pct:.2f}%)"
            val_rmse_str   += f" ({val_rmse_pct:.2f}%)"

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} RMSE {train_rmse_str} LCP {train_lcp:.2%} | "
            f"Val Loss {val_loss:.4f} RMSE {val_rmse_str} LCP {val_lcp:.2%} | "
            f"Epoch {epoch_time:.1f}s | Total {total_time/60:.1f}min"
        )

        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss, val_loss,
                train_rmse, val_rmse,
                RMSE_UNIT, RMSE_UNIT,
                train_rmse_pct, val_rmse_pct,
                train_lcp, val_lcp,
                epoch_time, total_time,
                train_iter_time, val_iter_time
            ])

        if SAVE_BEST and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Sauvegardé : {best_path} (val_loss={best_val:.4f})")

        if epoch % 5 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"rpm_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] Checkpoint : {save_path}")

        if epoch % SAVE_EVERY == 0:
            if export_train_pack is not None:
                src_b, tgt_b, transforms, fname0, true_R_b, true_t_b = export_train_pack
                export_step_by_step(epoch, VIS_TRAIN_DIR,
                                    src_b, tgt_b, transforms,
                                    fname0, train_dataset,
                                    true_R_b, true_t_b)

            if export_test_pack is not None:
                src_b, tgt_b, transforms, fname0, true_R_b, true_t_b = export_test_pack
                export_step_by_step(epoch, VIS_TEST_DIR,
                                    src_b, tgt_b, transforms,
                                    fname0, test_dataset,
                                    true_R_b, true_t_b)

    try:
        import plot_curves
        plot_curves.plot_training_results(save_dir=PLOT_DIR, show=False)
        print(f"[PLOTS] enregistrés dans {PLOT_DIR}")
    except Exception as e:
        print(f"[PLOTS] erreur génération plots: {e}")


if __name__ == "__main__":
    train()