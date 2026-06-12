"""
run_all.py
==========
Lance tout le pipeline de preparation des donnees en une seule commande :
  1. data_preparation.py    - nettoyage, encodage, normalisation
  2. federated_partition.py - creation des 8 hopitaux (IID + non-IID)
  3. visualize_distribution.py - generation des 4 graphiques

Usage:
    python run_all.py
"""

import sys
import os
import time

# Ajouter src/ au path Python
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

def separator(title):
    print("\n" + "=" * 60)
    print(f"  ETAPE : {title}")
    print("=" * 60)

def main():
    start_total = time.time()

    print("\n" + "#" * 60)
    print("#   Federated Heart Disease - Pipeline complet")
    print("#   8 hopitaux simules | IID + non-IID")
    print("#" * 60)

    # ── Etape 1 : Data Preparation ────────────────────────────────
    separator("1/3  Data Preparation")
    t0 = time.time()
    from data_preparation import prepare_dataset
    X, y, feature_names, scaler = prepare_dataset(save=True)
    print(f"  Temps : {time.time() - t0:.2f}s")

    # ── Etape 2 : Federated Partition ─────────────────────────────
    separator("2/3  Federated Partition (8 hopitaux)")
    t0 = time.time()
    from federated_partition import partition_data
    splits = partition_data()
    print(f"  Temps : {time.time() - t0:.2f}s")

    # ── Etape 3 : Visualisation ───────────────────────────────────
    separator("3/3  Generation des graphiques")
    t0 = time.time()
    from visualize_distribution import generate_all_figures
    fig_paths = generate_all_figures()
    print(f"  Temps : {time.time() - t0:.2f}s")

    # ── Resume final ──────────────────────────────────────────────
    elapsed = time.time() - start_total
    print("\n" + "#" * 60)
    print("#   Pipeline termine avec succes!")
    print(f"#   Temps total : {elapsed:.2f}s")
    print("#" * 60)

    print("\n  Fichiers produits :")
    print("    outputs/processed/X.npy              <- features (1025 x 23)")
    print("    outputs/processed/y.npy              <- labels   (1025,)")
    print("    outputs/clients_iid/client_*_X.npy   <- 8 hopitaux IID")
    print("    outputs/clients_noniid/client_*_X.npy <- 8 hopitaux non-IID")
    print("    outputs/figures/*.png                <- 4 graphiques\n")

    print("  Graphiques generes :")
    for p in fig_paths:
        print(f"    {os.path.basename(p)}")
    print()


if __name__ == "__main__":
    main()
