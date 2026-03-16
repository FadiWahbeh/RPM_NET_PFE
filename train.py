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

# SETTINGS

SAVE_BEST = True
BEST_NAME = "rpm_best.pth"

LCP_RATIO = 0.02
RMSE_UNIT = "m"
LOG_RMSE_NORM = True

# ICP only for EXPORT
USE_ICP_FOR_EXPORT = True
ICP_VOXEL = 0.10
ICP_MAX_CORR_FACTOR = 0.05
ICP_ITERS = 30
ICP_POINT_TO_PLANE = True

# Metrics speed
METRICS_MAX_BATCHES_TRAIN = 5
METRICS_MAX_BATCHES_VAL = 10

# Early stopping / rollback
EARLY_STOPPING = True
PATIENCE_EPOCHS = 25
ROLLBACK_ON_DEGRADE = True
DEGRADE_TOL = 1.10
ROLLBACK_LR_FACTOR = 0.5

# Exports
EXPORT_EVERY = 1

# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [BASE_DIR, os.path.join(BASE_DIR, "models"), os.path.join(BASE_DIR, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from matrics import compute_rmse_cloudcompare, compute_lcp_adaptive
from paired_dataset import PairedKinectLidarDataset

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

os.makedirs(VIS_TRAIN_DIR, exist_ok=True)
os.makedirs(VIS_TEST_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMS
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4
NUM_POINTS = 1024
NUM_ITERS = 5

# DataLoader perferences
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# LOG FILES 
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(OUTPUT_DIR, f"training_log_{RUN_STAMP}.csv")
LOG_LATEST = os.path.join(OUTPUT_DIR, "training_log_latest.csv")


# Helpers  
def batch_diag_from_tgt(tgt_b3n: torch.Tensor, eps=1e-9) -> torch.Tensor:
    pmin = tgt_b3n.amin(dim=2)
    pmax = tgt_b3n.amax(dim=2)
    diag = torch.norm(pmax - pmin, dim=1)
    return torch.clamp(diag, min=eps)

def rmse_percent_of_size(rmse_abs: float, diag: float) -> float:
    if diag <= 1e-12:
        return 0.0
    return 100.0 * (rmse_abs / diag)

def diag_np_from_points(points_np: np.ndarray) -> float:
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

def _to_o3d_pcd(points_np: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    return pcd

def icp_refine_o3d(src_np: np.ndarray, tgt_np: np.ndarray) -> np.ndarray:
    src = _to_o3d_pcd(src_np)
    tgt = _to_o3d_pcd(tgt_np)

    if ICP_VOXEL and ICP_VOXEL > 0:
        src = src.voxel_down_sample(float(ICP_VOXEL))
        tgt = tgt.voxel_down_sample(float(ICP_VOXEL))

    if len(src.points) < 30 or len(tgt.points) < 30:
        return src_np

    if ICP_POINT_TO_PLANE:
        radius = float(max(ICP_VOXEL * 2.0, 1e-6))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    tgt_pts = np.asarray(tgt.points)
    pmin = tgt_pts.min(axis=0)
    pmax = tgt_pts.max(axis=0)
    diag = float(np.linalg.norm(pmax - pmin)) if tgt_pts.shape[0] > 10 else 1.0
    max_corr = float(max(1e-6, ICP_MAX_CORR_FACTOR * diag))

    T0 = np.eye(4, dtype=np.float64)

    if ICP_POINT_TO_PLANE:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, T0, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(ICP_ITERS))
    )
    T = reg.transformation
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    return (src_np @ R.T) + t

def export_low(epoch_idx, base_dir, src_tensor, tgt_tensor, transforms):
    ep_dir = os.path.join(base_dir, f"Ep{epoch_idx:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    src0 = src_tensor[0].detach().cpu().numpy().T
    tgt0 = tgt_tensor[0].detach().cpu().numpy().T

    save_colored_single(tgt0, [0, 1, 0], os.path.join(ep_dir, "Target_low.ply"))
    save_colored_single(src0, [1, 0, 0], os.path.join(ep_dir, "Start_low.ply"))
    save_colored_pair(src0, [1, 0, 0], tgt0, [0, 1, 0], os.path.join(ep_dir, "Start_plus_Target_low.ply"))

    last_src_it = None
    for i, (r_it, t_it) in enumerate(transforms):
        src_it = transform_point_cloud_torch(src_tensor, r_it, t_it)[0].detach().cpu().numpy().T
        last_src_it = src_it
        save_colored_single(src_it, [0, 0, 1], os.path.join(ep_dir, f"Result_it{i+1}_low.ply"))
        save_colored_pair(src_it, [0, 0, 1], tgt0, [0, 1, 0], os.path.join(ep_dir, f"Result_it{i+1}_plus_Target_low.ply"))

    if USE_ICP_FOR_EXPORT and last_src_it is not None:
        src_icp = icp_refine_o3d(last_src_it, tgt0)
        save_colored_single(src_icp, [0, 1, 1], os.path.join(ep_dir, "Result_ICP_low.ply"))
        save_colored_pair(src_icp, [0, 1, 1], tgt0, [0, 1, 0], os.path.join(ep_dir, "Result_ICP_plus_Target_low.ply"))


def train():
    print("TRAIN Kinect->LiDAR (FAST) - unique CSV per run + latest alias")
    print(f"[LOG] {LOG_FILE}")

    # dataset
    train_dataset = PairedKinectLidarDataset(BASE_DIR, mode="train", num_points=NUM_POINTS, preload_cache=True)
    test_dataset  = PairedKinectLidarDataset(BASE_DIR, mode="test",  num_points=NUM_POINTS, preload_cache=True)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Dataset pairé vide. Vérifie noms identiques Kinect/LiDAR.")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )

    model = RPMNet(num_iterations=NUM_ITERS, use_dsc_lite=True, dsc_k=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(CHECKPOINT_DIR, BEST_NAME)
    best_state = None
    bad_epochs = 0

    # Write CSV header
    header = [
        "Epoch",
        "Train_Loss", "Val_Loss",
        "Train_RMSE", "Val_RMSE",
        "Train_RMSE_Unit", "Val_RMSE_Unit",
        "Train_RMSE_Pct", "Val_RMSE_Pct",
        "Train_LCP", "Val_LCP",
        "Epoch_Time_s", "Total_Time_s"
    ]
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    # update latest alias
    try:
        with open(LOG_LATEST, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["log_file"])
            csv.writer(f).writerow([os.path.basename(LOG_FILE)])
    except Exception:
        pass

    t_total0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        t_epoch0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        train_rmse_list, train_rmse_pct_list, train_lcp_list = [], [], []
        export_pack = None

        for bi, (src, tgt, _true_R, _true_t, fnames) in enumerate(train_loader):
            src = src.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # rpmnet returns 3 values when return_timings=False
            R_pred, t_pred, transforms = model(src, tgt, return_timings=False)

            diag = batch_diag_from_tgt(tgt)
            diag_b = diag.view(-1, 1, 1)

            loss = 0.0
            n_iters = len(transforms)
            for i, (r_iter, t_iter) in enumerate(transforms):
                w = 1.0 / (2 ** (n_iters - 1 - i))
                src_iter = transform_point_cloud_torch(src, r_iter, t_iter)
                err = (src_iter - tgt) / diag_b
                loss = loss + w * mse_loss(err, torch.zeros_like(err))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())

            if export_pack is None:
                export_pack = (src.detach(), tgt.detach(), transforms)

            # Metrics only on subset
            if bi < METRICS_MAX_BATCHES_TRAIN:
                with torch.no_grad():
                    src_final = transform_point_cloud_torch(src, R_pred, t_pred)
                    for b in range(src.size(0)):
                        p_pred = src_final[b].transpose(0, 1)
                        p_tgt = tgt[b].transpose(0, 1)

                        rmse_abs = compute_rmse_cloudcompare(p_pred, p_tgt)
                        train_rmse_list.append(rmse_abs)

                        if LOG_RMSE_NORM:
                            d = diag_np_from_points(p_tgt.detach().cpu().numpy())
                            train_rmse_pct_list.append(rmse_percent_of_size(rmse_abs, d))

                        train_lcp_list.append(compute_lcp_adaptive(p_pred, p_tgt, ratio=LCP_RATIO))

        train_loss = total_loss / max(1, len(train_loader))
        train_rmse = float(np.mean(train_rmse_list)) if train_rmse_list else 0.0
        train_rmse_pct = float(np.mean(train_rmse_pct_list)) if train_rmse_pct_list else 0.0
        train_lcp = float(np.mean(train_lcp_list)) if train_lcp_list else 0.0

        model.eval()
        val_loss_acc = 0.0
        val_rmse_list, val_rmse_pct_list, val_lcp_list = [], [], []

        with torch.no_grad():
            for bi, (src, tgt, _true_R, _true_t, fnames) in enumerate(test_loader):
                src = src.to(DEVICE, non_blocking=True)
                tgt = tgt.to(DEVICE, non_blocking=True)

                R_pred, t_pred, transforms = model(src, tgt, return_timings=False)

                diag = batch_diag_from_tgt(tgt)
                diag_b = diag.view(-1, 1, 1)

                last_r, last_t = transforms[-1]
                src_final_val = transform_point_cloud_torch(src, last_r, last_t)
                err = (src_final_val - tgt) / diag_b
                val_loss_acc += float(mse_loss(err, torch.zeros_like(err)).item())

                if bi < METRICS_MAX_BATCHES_VAL:
                    for b in range(src.size(0)):
                        p_pred = src_final_val[b].transpose(0, 1)
                        p_tgt = tgt[b].transpose(0, 1)

                        rmse_abs = compute_rmse_cloudcompare(p_pred, p_tgt)
                        val_rmse_list.append(rmse_abs)

                        if LOG_RMSE_NORM:
                            d = diag_np_from_points(p_tgt.detach().cpu().numpy())
                            val_rmse_pct_list.append(rmse_percent_of_size(rmse_abs, d))

                        val_lcp_list.append(compute_lcp_adaptive(p_pred, p_tgt, ratio=LCP_RATIO))

        val_loss = val_loss_acc / max(1, len(test_loader))
        val_rmse = float(np.mean(val_rmse_list)) if val_rmse_list else 0.0
        val_rmse_pct = float(np.mean(val_rmse_pct_list)) if val_rmse_pct_list else 0.0
        val_lcp = float(np.mean(val_lcp_list)) if val_lcp_list else 0.0

        scheduler.step(val_loss)

        t_epoch1 = time.perf_counter()
        epoch_time = t_epoch1 - t_epoch0
        total_time = t_epoch1 - t_total0

        train_rmse_str = f"{train_rmse:.4f}{RMSE_UNIT}" + (f" ({train_rmse_pct:.2f}%)" if LOG_RMSE_NORM else "")
        val_rmse_str = f"{val_rmse:.4f}{RMSE_UNIT}" + (f" ({val_rmse_pct:.2f}%)" if LOG_RMSE_NORM else "")

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} RMSE {train_rmse_str} LCP {train_lcp:.2%} | "
            f"Val Loss {val_loss:.4f} RMSE {val_rmse_str} LCP {val_lcp:.2%} | "
            f"Epoch {epoch_time:.1f}s | Total {total_time/60:.1f}min"
        )

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss, val_loss,
                train_rmse, val_rmse,
                RMSE_UNIT, RMSE_UNIT,
                train_rmse_pct, val_rmse_pct,
                train_lcp, val_lcp,
                epoch_time, total_time
            ])

        # BEST
        if SAVE_BEST and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(f"[BEST] Saved {best_path} (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1

        # Rollback
        if ROLLBACK_ON_DEGRADE and best_state is not None and val_loss > DEGRADE_TOL * best_val:
            print(f"[ROLLBACK] val_loss={val_loss:.4f} > {DEGRADE_TOL:.2f}*best={best_val:.4f} -> reload best, lr*= {ROLLBACK_LR_FACTOR}")
            model.load_state_dict(best_state)
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * ROLLBACK_LR_FACTOR
            bad_epochs = 0

        # Early stop
        if EARLY_STOPPING and bad_epochs >= PATIENCE_EPOCHS:
            print(f"[EARLY STOP] no improvement for {PATIENCE_EPOCHS} epochs.")
            break

        # Exports
        if export_pack is not None and epoch % EXPORT_EVERY == 0:
            src_b, tgt_b, tr = export_pack
            export_low(epoch, VIS_TRAIN_DIR, src_b, tgt_b, tr)

    print(f"[DONE] Log saved: {LOG_FILE}")
    print(f"[DONE] Latest pointer: {LOG_LATEST}")


if __name__ == "__main__":
    train()