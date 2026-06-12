"""
federated_partition.py
======================
Task 1 – Data Preparation & Hospital Simulation
Federated Heart Disease Project

Creates two types of federated data partitions across N_CLIENTS hospitals:
  - IID   : random shuffle, equal split
  - non-IID: Dirichlet(alpha) distribution to simulate realistic class imbalance

Outputs saved to:
  outputs/clients_iid/client_i_{X,y}.npy
  outputs/clients_noniid/client_i_{X,y}.npy
"""

import os
import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
N_CLIENTS      = 8        # number of simulated hospitals
DIRICHLET_ALPHA = 0.5     # lower = more heterogeneous (0.1 very skewed, 1.0 mild)
RANDOM_SEED    = 42

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR  = os.path.join(BASE_DIR, "outputs", "processed")
IID_DIR        = os.path.join(BASE_DIR, "outputs", "clients_iid")
NONIID_DIR     = os.path.join(BASE_DIR, "outputs", "clients_noniid")

for d in [IID_DIR, NONIID_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_processed() -> tuple[np.ndarray, np.ndarray]:
    """Load the preprocessed X, y arrays produced by data_preparation.py."""
    X_path = os.path.join(PROCESSED_DIR, "X.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")
    if not os.path.exists(X_path):
        raise FileNotFoundError(
            "Processed data not found. Run src/data_preparation.py first."
        )
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y


def print_partition_stats(splits: list[tuple], mode: str) -> None:
    """Pretty-print per-client statistics."""
    print(f"\n{'='*62}")
    print(f"  Partition mode: {mode}")
    print(f"  {'Hospital':<12} {'Samples':>8} {'Sick (1)':>10} {'Healthy (0)':>12} {'% Sick':>8}")
    print(f"  {'-'*58}")
    for i, (Xi, yi) in enumerate(splits):
        n       = len(yi)
        n_sick  = int(yi.sum())
        n_well  = n - n_sick
        pct     = n_sick / n * 100 if n > 0 else 0
        print(f"  Hospital {i+1:<3}  {n:>8} {n_sick:>10} {n_well:>12} {pct:>7.1f}%")
    print(f"{'='*62}\n")


def save_splits(splits: list[tuple], out_dir: str) -> None:
    """Save each client's (X, y) arrays to disk."""
    for i, (Xi, yi) in enumerate(splits):
        np.save(os.path.join(out_dir, f"client_{i}_X.npy"), Xi)
        np.save(os.path.join(out_dir, f"client_{i}_y.npy"), yi)
    print(f"  [OK] Saved {len(splits)} clients -> {out_dir}")


# ── IID Partition ──────────────────────────────────────────────────────────────

def iid_partition(X: np.ndarray, y: np.ndarray, n_clients: int = N_CLIENTS,
                  seed: int = RANDOM_SEED) -> list[tuple]:
    """
    Randomly shuffle and split data equally across clients.
    Each client receives approximately the same number of samples
    with roughly the same class distribution as the global dataset.
    """
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(y))
    chunks = np.array_split(idx, n_clients)
    splits = [(X[c], y[c]) for c in chunks]
    return splits


# ── non-IID Partition (Dirichlet) ──────────────────────────────────────────────

def noniid_partition(X: np.ndarray, y: np.ndarray, n_clients: int = N_CLIENTS,
                     alpha: float = DIRICHLET_ALPHA,
                     seed: int = RANDOM_SEED) -> list[tuple]:
    """
    Dirichlet-based non-IID partition.

    For each class c, draw a Dirichlet(alpha) sample that determines what
    fraction of class-c samples goes to each hospital.
    Small alpha → highly heterogeneous (some hospitals see mostly one class).
    alpha=0.5 is a standard benchmark choice in federated learning literature.
    """
    rng        = np.random.default_rng(seed)
    classes    = np.unique(y)
    client_idx = [[] for _ in range(n_clients)]

    for c in classes:
        class_idx  = np.where(y == c)[0]
        rng.shuffle(class_idx)

        # Proportions for this class across clients
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)

        # Convert proportions to cumulative counts
        counts = (proportions * len(class_idx)).astype(int)
        # Fix rounding so total == len(class_idx)
        counts[-1] = len(class_idx) - counts[:-1].sum()

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            client_idx[client_id].extend(class_idx[start:end].tolist())
            start = end

    # Shuffle within each client
    splits = []
    for idx in client_idx:
        idx = np.array(idx)
        rng.shuffle(idx)
        splits.append((X[idx], y[idx]))

    return splits


# ── Main ───────────────────────────────────────────────────────────────────────

def partition_data(n_clients: int = N_CLIENTS,
                   alpha: float = DIRICHLET_ALPHA,
                   seed: int = RANDOM_SEED) -> dict:
    """
    Run both IID and non-IID partitions and save results.

    Returns
    -------
    dict with keys 'iid' and 'noniid', each a list of (X_i, y_i) tuples.
    """
    print(f"\n{'='*62}")
    print(f"  Federated Partition - {n_clients} Hospitals")
    print(f"  IID split  +  non-IID Dirichlet(alpha={alpha})")
    print(f"{'='*62}")

    X, y = load_processed()
    print(f"\n  Loaded processed data: X={X.shape}, y={y.shape}")

    # IID
    iid_splits = iid_partition(X, y, n_clients=n_clients, seed=seed)
    print_partition_stats(iid_splits, mode="IID")
    save_splits(iid_splits, IID_DIR)

    # non-IID
    noniid_splits = noniid_partition(X, y, n_clients=n_clients, alpha=alpha, seed=seed)
    print_partition_stats(noniid_splits, mode=f"non-IID (Dirichlet alpha={alpha})")
    save_splits(noniid_splits, NONIID_DIR)

    print(f"\n{'='*62}")
    print("  Partitioning complete!")
    print(f"{'='*62}\n")

    return {"iid": iid_splits, "noniid": noniid_splits}


if __name__ == "__main__":
    partition_data()
