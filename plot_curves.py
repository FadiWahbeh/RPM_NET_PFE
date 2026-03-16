import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


WANTED_COLS = [
    "Epoch",
    "Train_Loss", "Val_Loss",
    "Train_RMSE", "Val_RMSE",
    "Train_RMSE_Pct", "Val_RMSE_Pct",
    "Train_LCP", "Val_LCP",
    "Epoch_Time_s", "Total_Time_s",
    "Train_IterTimeMean_s", "Val_IterTimeMean_s",
    "Train_RMSE_Unit", "Val_RMSE_Unit",
]


def robust_read_training_log(path: str) -> pd.DataFrame:
    """Lit training_log.csv même si le nombre de colonnes varie selon les lignes."""
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        raw = list(csv.reader(f))

    if len(raw) < 2:
        return pd.DataFrame()

    header = raw[0]
    rows = raw[1:]

    max_cols = max(len(header), max((len(r) for r in rows if len(r) > 0), default=len(header)))

    if len(header) < max_cols:
        header = header + [f"extra_{i}" for i in range(len(header), max_cols)]

    fixed = []
    dropped = 0
    for r in rows:
        if len(r) == 0:
            dropped += 1
            continue
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        fixed.append(r)

    df = pd.DataFrame(fixed, columns=header)

    # convert numeric columns
    for c in df.columns:
        if c in ("Train_RMSE_Unit", "Val_RMSE_Unit"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Epoch" in df.columns:
        df = df[df["Epoch"].notna()].copy()
        df["Epoch"] = df["Epoch"].astype(int)

    # keep useful subset
    cols = [c for c in WANTED_COLS if c in df.columns]
    if cols:
        df = df[cols]

    print(f"[plot_curves] Loaded rows={len(df)} cols={len(df.columns)} (ignored empty lines={dropped})")
    return df


def save_tables(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    if "Epoch" in df.columns:
        df = df.sort_values("Epoch")

    # last 30
    df.tail(30).to_csv(os.path.join(out_dir, "table_last_30.csv"), index=False)

    # best val loss
    if "Val_Loss" in df.columns and df["Val_Loss"].notna().any():
        df.loc[df["Val_Loss"].idxmin()].to_frame().T.to_csv(
            os.path.join(out_dir, "table_best_val_loss.csv"), index=False
        )
    else:
        pd.DataFrame({"info": ["Val_Loss missing"]}).to_csv(
            os.path.join(out_dir, "table_best_val_loss.csv"), index=False
        )

    # best val rmse
    if "Val_RMSE" in df.columns and df["Val_RMSE"].notna().any():
        df.loc[df["Val_RMSE"].idxmin()].to_frame().T.to_csv(
            os.path.join(out_dir, "table_best_val_rmse.csv"), index=False
        )
    else:
        pd.DataFrame({"info": ["Val_RMSE missing"]}).to_csv(
            os.path.join(out_dir, "table_best_val_rmse.csv"), index=False
        )

    # summary txt
    lines = []
    lines.append("=== SUMMARY ===")
    lines.append(f"Rows: {len(df)}")
    if "Epoch" in df.columns and len(df) > 0:
        lines.append(f"Epoch range: {int(df['Epoch'].min())} -> {int(df['Epoch'].max())}")

    if "Val_Loss" in df.columns and df["Val_Loss"].notna().any():
        e = int(df.loc[df["Val_Loss"].idxmin()]["Epoch"])
        lines.append(f"Best Val_Loss: {df['Val_Loss'].min():.6f} at epoch {e}")

    if "Val_RMSE" in df.columns and df["Val_RMSE"].notna().any():
        e = int(df.loc[df["Val_RMSE"].idxmin()]["Epoch"])
        lines.append(f"Best Val_RMSE: {df['Val_RMSE'].min():.6f} at epoch {e}")

    if "Val_LCP" in df.columns and df["Val_LCP"].notna().any():
        e = int(df.loc[df["Val_LCP"].idxmax()]["Epoch"])
        lines.append(f"Best Val_LCP: {df['Val_LCP'].max():.6f} at epoch {e}")

    lines.append("\n=== LAST 10 ===")
    lines.append(df.tail(10).to_string(index=False))

    with open(os.path.join(out_dir, "table_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[plot_curves] Tables saved in:", out_dir)


def plot_training_results(save_dir=None, show=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    log_file = os.path.join(output_dir, "training_log.csv")

    if save_dir is None:
        save_dir = os.path.join(output_dir, "Plot")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(log_file):
        print("Fichier log introuvable:", log_file)
        return

    df = robust_read_training_log(log_file)
    if df.empty:
        print("training_log.csv est vide ou illisible.")
        return

    # make tables now
    save_tables(df, save_dir)

    def has(cols):
        return all(c in df.columns for c in cols)

    # LOSS
    if has(["Epoch", "Train_Loss", "Val_Loss"]):
        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["Train_Loss"], label="Train Loss")
        plt.plot(df["Epoch"], df["Val_Loss"], label="Test Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss: Train vs Test")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_train_vs_test.png"))
        if show:
            plt.show()
        plt.close()

    # RMSE
    if has(["Epoch", "Train_RMSE", "Val_RMSE"]):
        unit = ""
        if "Train_RMSE_Unit" in df.columns and df["Train_RMSE_Unit"].notna().any():
            unit = str(df["Train_RMSE_Unit"].dropna().iloc[0])
            if unit and not unit.startswith(" "):
                unit = " " + unit

        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["Train_RMSE"], label=f"Train RMSE{unit}")
        plt.plot(df["Epoch"], df["Val_RMSE"], label=f"Test RMSE{unit}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel(f"RMSE{unit}")
        plt.title("RMSE: Train vs Test")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rmse_train_vs_test.png"))
        if show:
            plt.show()
        plt.close()

    # RMSE %
    if has(["Epoch", "Train_RMSE_Pct", "Val_RMSE_Pct"]):
        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["Train_RMSE_Pct"], label="Train RMSE (%)")
        plt.plot(df["Epoch"], df["Val_RMSE_Pct"], label="Test RMSE (%)", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (% of size)")
        plt.title("RMSE (% size): Train vs Test")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rmse_percent_train_vs_test.png"))
        if show:
            plt.show()
        plt.close()

    # LCP
    if has(["Epoch", "Train_LCP", "Val_LCP"]):
        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["Train_LCP"], label="Train LCP")
        plt.plot(df["Epoch"], df["Val_LCP"], label="Test LCP", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("LCP")
        plt.title("LCP: Train vs Test")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "lcp_train_vs_test.png"))
        if show:
            plt.show()
        plt.close()

    print(f"[plot_curves] Done. Tables + plots saved in: {save_dir}")


if __name__ == "__main__":
    plot_training_results()