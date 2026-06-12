"""
visualize_distribution.py
=========================
Task 1 – Data Preparation & Hospital Simulation
Federated Heart Disease Project

Generates 4 publication-quality figures saved to outputs/figures/:
  1. global_class_distribution.png  — overall class balance
  2. samples_per_client.png         — samples per hospital (IID vs non-IID)
  3. class_dist_iid.png             — per-client class distribution (IID)
  4. class_dist_noniid.png          — per-client class distribution (non-IID)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts & notebooks)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IID_DIR     = os.path.join(BASE_DIR, "outputs", "clients_iid")
NONIID_DIR  = os.path.join(BASE_DIR, "outputs", "clients_noniid")
PROC_DIR    = os.path.join(BASE_DIR, "outputs", "processed")
FIG_DIR     = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
COLOR_SICK    = "#E05C5C"   # red-ish for sick (target=1)
COLOR_HEALTHY = "#5B9BD5"   # blue-ish for healthy (target=0)
COLOR_IID     = "#4CAF84"   # green for IID bars
COLOR_NONIID  = "#F4A261"   # orange for non-IID bars
BG_COLOR      = "#F8F9FA"
GRID_COLOR    = "#DEE2E6"

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.facecolor"    : BG_COLOR,
    "figure.facecolor"  : "white",
    "axes.grid"         : True,
    "grid.color"        : GRID_COLOR,
    "grid.linewidth"    : 0.8,
    "axes.edgecolor"    : "#CED4DA",
})

N_CLIENTS = 8
HOSPITAL_LABELS = [f"H{i+1}" for i in range(N_CLIENTS)]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_splits(directory: str, n_clients: int = N_CLIENTS) -> list[np.ndarray]:
    """Load y arrays for each client."""
    splits = []
    for i in range(n_clients):
        y = np.load(os.path.join(directory, f"client_{i}_y.npy"))
        splits.append(y)
    return splits


# ── Figure 1: Global class distribution ───────────────────────────────────────

def plot_global_class_distribution():
    y = np.load(os.path.join(PROC_DIR, "y.npy"))
    n_sick    = int(y.sum())
    n_healthy = int((y == 0).sum())
    total     = len(y)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        ["Healthy (0)", "Sick (1)"],
        [n_healthy, n_sick],
        color=[COLOR_HEALTHY, COLOR_SICK],
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, count in zip(bars, [n_healthy, n_sick]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count}\n({count/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax.set_title("Global Class Distribution\n(Heart Disease Dataset — 1025 patients)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_ylim(0, max(n_sick, n_healthy) * 1.18)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "global_class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {os.path.basename(path)}")
    return path


# ── Figure 2: Samples per client — IID vs non-IID ─────────────────────────────

def plot_samples_per_client():
    iid_splits    = load_splits(IID_DIR)
    noniid_splits = load_splits(NONIID_DIR)

    iid_counts    = [len(y) for y in iid_splits]
    noniid_counts = [len(y) for y in noniid_splits]

    x      = np.arange(N_CLIENTS)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, iid_counts,    width, color=COLOR_IID,    label="IID",
                edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + width/2, noniid_counts, width, color=COLOR_NONIID, label="non-IID",
                edgecolor="white", linewidth=1.2)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Number of Samples per Hospital\n(IID vs non-IID Partition)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Hospital", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(HOSPITAL_LABELS)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "samples_per_client.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {os.path.basename(path)}")
    return path


# ── Figure 3 & 4: Stacked bars per client ──────────────────────────────────────

def plot_class_dist_per_client(splits: list[np.ndarray], mode: str, filename: str):
    sick_counts    = [int(y.sum()) for y in splits]
    healthy_counts = [int((y == 0).sum()) for y in splits]
    totals         = [s + h for s, h in zip(sick_counts, healthy_counts)]

    x = np.arange(N_CLIENTS)
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Per-Hospital Class Distribution — {mode}",
                 fontsize=14, fontweight="bold", y=1.01)

    # — Left: absolute counts —
    ax_abs.bar(x, healthy_counts, color=COLOR_HEALTHY, label="Healthy (0)", edgecolor="white")
    ax_abs.bar(x, sick_counts,    color=COLOR_SICK,    label="Sick (1)",
               bottom=healthy_counts, edgecolor="white")
    ax_abs.set_title("Absolute Counts", fontsize=12)
    ax_abs.set_xlabel("Hospital")
    ax_abs.set_ylabel("Samples")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(HOSPITAL_LABELS)
    ax_abs.legend(framealpha=0.9)

    for i, total in enumerate(totals):
        ax_abs.text(i, total + 1.5, str(total),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # — Right: percentage —
    sick_pct    = [s / t * 100 if t > 0 else 0 for s, t in zip(sick_counts, totals)]
    healthy_pct = [100 - p for p in sick_pct]

    ax_pct.bar(x, healthy_pct, color=COLOR_HEALTHY, label="Healthy (0)", edgecolor="white")
    ax_pct.bar(x, sick_pct,    color=COLOR_SICK,    label="Sick (1)",
               bottom=healthy_pct, edgecolor="white")

    # Annotate % sick
    for i, pct in enumerate(sick_pct):
        y_pos = healthy_pct[i] + sick_pct[i] / 2
        ax_pct.text(i, y_pos, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    ax_pct.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.5, label="50% line")
    ax_pct.set_title("Class Percentage", fontsize=12)
    ax_pct.set_xlabel("Hospital")
    ax_pct.set_ylabel("Percentage (%)")
    ax_pct.set_ylim(0, 105)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(HOSPITAL_LABELS)
    ax_pct.legend(framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {os.path.basename(path)}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_all_figures() -> list[str]:
    """Generate all 4 figures and return their paths."""
    print(f"\n{'='*60}")
    print(f"  Generating Distribution Figures")
    print(f"{'='*60}\n")

    paths = []
    paths.append(plot_global_class_distribution())
    paths.append(plot_samples_per_client())

    iid_splits    = load_splits(IID_DIR)
    noniid_splits = load_splits(NONIID_DIR)

    paths.append(plot_class_dist_per_client(
        iid_splits, mode="IID", filename="class_dist_iid.png"
    ))
    paths.append(plot_class_dist_per_client(
        noniid_splits, mode="non-IID (Dirichlet alpha=0.5)", filename="class_dist_noniid.png"
    ))

    print(f"  [OK] Saved to: {FIG_DIR}")
    print(f"{'='*60}\n")
    return paths


if __name__ == "__main__":
    generate_all_figures()
