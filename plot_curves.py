import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_results(save_dir=None, show=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    log_file = os.path.join(output_dir, "training_log.csv")

    if save_dir is None:
        save_dir = os.path.join(output_dir, "Plot")

    if not os.path.exists(log_file):
        print("Fichier log introuvable. Lance train.py d'abord.")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(log_file)

    if df.shape[0] == 0:
        print("training_log.csv est vide. Laisse train.py écrire quelques epochs.")
        return

    def has_cols(cols):
        return all(c in df.columns for c in cols)

    # 1) LOSS
    if has_cols(["Epoch", "Train_Loss", "Val_Loss"]):
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
    else:
        print("[WARN] Colonnes loss manquantes dans le CSV.")

    # 2) RMSE
    if has_cols(["Epoch", "Train_RMSE", "Val_RMSE"]):
        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["Train_RMSE"], label="Train RMSE")
        plt.plot(df["Epoch"], df["Val_RMSE"], label="Test RMSE", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("RMSE: Train vs Test")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rmse_train_vs_test.png"))
        if show:
            plt.show()
        plt.close()
    else:
        print("[WARN] Colonnes RMSE manquantes dans le CSV.")

    # 3) LCP
    if has_cols(["Epoch", "Train_LCP", "Val_LCP"]):
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
    else:
        print("[WARN] Colonnes LCP manquantes dans le CSV.")

    # 4) TIMINGS (optionnel)
    if "Epoch" in df.columns and "Epoch_Time_s" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["Epoch"], df["Epoch_Time_s"], label="Epoch time (s)")

        if "Train_IterTimeMean_s" in df.columns:
            plt.plot(df["Epoch"], df["Train_IterTimeMean_s"], label="Mean RPM iter time TRAIN (s)")
        if "Val_IterTimeMean_s" in df.columns:
            plt.plot(df["Epoch"], df["Val_IterTimeMean_s"], label="Mean RPM iter time TEST (s)")

        plt.xlabel("Epoch")
        plt.ylabel("Seconds")
        plt.title("Timings")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "timings.png"))
        if show:
            plt.show()
        plt.close()

    print(f"Plots sauvegardés dans: {save_dir}")


if __name__ == "__main__":
    plot_training_results()
