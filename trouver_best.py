import os
import csv

# Chemin vers ton fichier log
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "outputs", "training_log.csv")

def trouver_meilleure_epoque():
    if not os.path.exists(LOG_FILE):
        print("Erreur : Fichier training_log.csv introuvable.")
        return

    best_loss = float('inf')
    best_epoch = -1
    best_lcp = 0.0

    print(f"Lecture de {LOG_FILE} ...")
    
    with open(LOG_FILE, "r", encoding="utf-8") as f: # Ajout utf-8 au cas où
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Erreur : Le fichier CSV est vide.")
            return

        # Détection des colonnes
        try:
            idx_epoch = header.index("Epoch")
            idx_val_loss = header.index("Val_Loss")
            # idx_val_lcp = header.index("Val_LCP") # On enlève pour être sûr
            
            # Si Val_LCP existe, on le prend, sinon on ignore
            idx_val_lcp = -1
            if "Val_LCP" in header:
                idx_val_lcp = header.index("Val_LCP")

        except ValueError:
            idx_epoch = 0
            idx_val_loss = 2
            idx_val_lcp = 6

        for row in reader:
            if not row: continue
            try:
                epoch = int(row[idx_epoch])
                val_loss = float(row[idx_val_loss])
                
                # Ignore les 0.0 absolus (bugs précédents)
                if val_loss < 0.00001:
                    continue
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if idx_val_lcp != -1 and len(row) > idx_val_lcp:
                        best_lcp = float(row[idx_val_lcp])
            except ValueError:
                continue

    if best_epoch != -1:
        print("\n" + "="*40)
        print(f"MEILLEURE EPOQUE TROUVEE : {best_epoch}")
        print(f"Val Loss : {best_loss:.4f}")
        print(f"Val LCP  : {best_lcp:.2%}")
        print("="*40)
        print(f"\nTu dois utiliser le fichier : outputs/checkpoints/rpm_epoch_{best_epoch}.pth")
        
        ckpt_path = os.path.join(BASE_DIR, "outputs", "checkpoints", f"rpm_epoch_{best_epoch}.pth")
        if os.path.exists(ckpt_path):
            print("   (Le fichier existe bien !)")
        else:
            print("   (Attention : Le fichier .pth semble avoir ete supprime)")
    else:
        print("Aucune donnee valide trouvee.")

if __name__ == "__main__":
    trouver_meilleure_epoque()