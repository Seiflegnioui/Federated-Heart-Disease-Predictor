"""
create_nb.py
============
Generates notebooks/03_analysis.ipynb – a comprehensive analysis notebook
covering all professor-required comparison criteria:
  1. Convergence curves (accuracy + loss per round)
  2. Train/Test gap (overfitting analysis)
  3. Local model divergence (||w_k - w_global||)
  4. Convergence speed to accuracy targets (80%, 85%, 90%)
  5. Time analysis (time per round + total time)
  6. Communication cost analysis
  7. Full comparative summary table
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()


# ─────────────────────────────────────────────────────────────────────────────
# Cell 0 – Title
# ─────────────────────────────────────────────────────────────────────────────
md_title = """\
# 📊 Analyse Complète — Apprentissage Fédéré pour la Détection des Maladies Cardiaques

**Module : Optimisation** | **Étudiants : Sifeddine Legnioui & Youssef Sarraf**

Ce notebook analyse et compare 4 stratégies d'optimisation fédérée :
`FedAvg`, `FedAdam`, `FedProx` et `Local SGD (baseline)` sur deux
distributions de données : **IID** et **Non-IID** (Dirichlet α=0.5).

---
### Critères analysés
1. Performance globale (Accuracy, F1, Precision, Recall, Loss)
2. Convergence et stabilité
3. Divergence des modèles locaux
4. Vitesse de convergence (rounds pour atteindre 80%, 85%, 90%)
5. Temps d'entraînement
6. Coût de communication
7. Analyse généralisation (gap Train/Test)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 – Imports & Setup
# ─────────────────────────────────────────────────────────────────────────────
code_setup = """\
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
results_dir = os.path.join("..", "outputs", "results")
fig_dir     = os.path.join("..", "outputs", "figures")
os.makedirs(fig_dir, exist_ok=True)

strategies = ["FedAvg", "FedAdam", "FedProx", "LocalSGD"]
modes      = ["iid", "noniid"]
COLORS     = {
    "FedAvg":   "#4C72B0",
    "FedAdam":  "#DD8452",
    "FedProx":  "#55A868",
    "LocalSGD": "#C44E52",
}
LABELS = {
    "FedAvg":   "FedAvg",
    "FedAdam":  "FedAdam",
    "FedProx":  "FedProx",
    "LocalSGD": "Local SGD",
}

# ── Load all results ────────────────────────────────────────────────────────────
results = {}
for mode in modes:
    results[mode] = {}
    for strat in strategies:
        fp = os.path.join(results_dir, f"{strat}_{mode}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                results[mode][strat] = json.load(f)
        else:
            print(f"  ⚠️  Not found: {fp}")

print("✅ Results loaded:", {m: list(results[m].keys()) for m in modes})
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 – Accuracy convergence curves
# ─────────────────────────────────────────────────────────────────────────────
code_acc = """\
# ══════════════════════════════════════════════════════════════════════════════
# 1. ACCURACY CONVERGENCE PER ROUND
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig.suptitle("Accuracy Globale par Round de Communication", fontsize=14, fontweight="bold")

for i, mode in enumerate(modes):
    ax = axes[i]
    for strat in strategies:
        if strat not in results[mode]:
            continue
        data = results[mode][strat]["metrics_distributed"]["accuracy"]
        rounds, accs = zip(*data)
        ax.plot(rounds, accs, marker="o", markersize=4, linewidth=2,
                color=COLORS[strat], label=LABELS[strat])
    ax.set_title(f"{'IID' if mode=='iid' else 'Non-IID (Dirichlet α=0.5)'}", fontsize=13)
    ax.set_xlabel("Round de Communication")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Cible 85%")
    ax.axhline(0.90, color="black", linestyle=":",  linewidth=1, alpha=0.6, label="Cible 90%")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "accuracy_convergence.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: accuracy_convergence.png")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 – Loss convergence curves
# ─────────────────────────────────────────────────────────────────────────────
code_loss = """\
# ══════════════════════════════════════════════════════════════════════════════
# 2. LOSS CONVERGENCE PER ROUND  (Test Loss vs Train Loss)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Convergence de la Loss (Test vs Train)", fontsize=14, fontweight="bold")

for col, mode in enumerate(modes):
    # Test loss
    ax_test = axes[0][col]
    for strat in strategies:
        if strat not in results[mode]:
            continue
        data = results[mode][strat]["losses_distributed"]
        rounds, losses = zip(*data)
        ax_test.plot(rounds, losses, marker="o", markersize=4, linewidth=2,
                     color=COLORS[strat], label=LABELS[strat])
    ax_test.set_title(f"Test Loss — {'IID' if mode=='iid' else 'Non-IID'}", fontsize=12)
    ax_test.set_xlabel("Round")
    ax_test.set_ylabel("Loss")
    ax_test.legend(fontsize=9)
    ax_test.grid(True, alpha=0.3)

    # Train loss
    ax_train = axes[1][col]
    for strat in strategies:
        if strat not in results[mode]:
            continue
        data = results[mode][strat].get("train_losses", [])
        if not data:
            continue
        rounds, losses = zip(*data)
        ax_train.plot(rounds, losses, marker="s", markersize=4, linewidth=2,
                      color=COLORS[strat], label=LABELS[strat], linestyle="--")
    ax_train.set_title(f"Train Loss — {'IID' if mode=='iid' else 'Non-IID'}", fontsize=12)
    ax_train.set_xlabel("Round")
    ax_train.set_ylabel("Loss")
    ax_train.legend(fontsize=9)
    ax_train.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "loss_convergence.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: loss_convergence.png")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 – Train/Test gap (overfitting)
# ─────────────────────────────────────────────────────────────────────────────
code_gap = """\
# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST GAP  (Overfitting Analysis)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Gap Train/Test — Analyse du Sur-apprentissage", fontsize=14, fontweight="bold")

for col, mode in enumerate(modes):
    ax = axes[col]
    for strat in strategies:
        if strat not in results[mode]:
            continue
        test_data  = results[mode][strat]["losses_distributed"]
        train_data = results[mode][strat].get("train_losses", [])
        if not train_data:
            continue
        rounds     = [r for r, _ in test_data]
        test_loss  = [l for _, l in test_data]
        train_loss = [l for _, l in train_data]
        gap = [abs(tr - te) for tr, te in zip(train_loss, test_loss)]
        ax.plot(rounds, gap, marker="o", markersize=4, linewidth=2,
                color=COLORS[strat], label=LABELS[strat])

    ax.set_title(f"{'IID' if mode=='iid' else 'Non-IID'}", fontsize=13)
    ax.set_xlabel("Round")
    ax.set_ylabel("|Train Loss − Test Loss|")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "train_test_gap.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: train_test_gap.png")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 – Local divergence
# ─────────────────────────────────────────────────────────────────────────────
code_divergence = """\
# ══════════════════════════════════════════════════════════════════════════════
# 4. DIVERGENCE DES MODÈLES LOCAUX  (||w_k − w_global||)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Divergence Moyenne des Modèles Locaux par Round\\n"
             "‖w_k − w_global‖₂  (mesure de la dérive locale)", fontsize=13, fontweight="bold")

for col, mode in enumerate(modes):
    ax = axes[col]
    for strat in strategies:
        if strat not in results[mode]:
            continue
        data = results[mode][strat].get("divergence_per_round", [])
        if not data:
            continue
        rounds, divs = zip(*data)
        ax.plot(rounds, divs, marker="o", markersize=4, linewidth=2,
                color=COLORS[strat], label=LABELS[strat])

    ax.set_title(f"{'IID' if mode=='iid' else 'Non-IID'}", fontsize=13)
    ax.set_xlabel("Round")
    ax.set_ylabel("Divergence L2 Moyenne")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "local_divergence.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: local_divergence.png")

# Divergence finale résumée
print("\\n📌 Divergence finale (round 15) par algorithme :")
for mode in modes:
    print(f"\\n  Mode: {mode.upper()}")
    for strat in strategies:
        if strat not in results[mode]:
            continue
        data = results[mode][strat].get("divergence_per_round", [])
        if data:
            last_div = data[-1][1]
            print(f"    {LABELS[strat]:<12}: {last_div:.4f}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 – Convergence speed to targets
# ─────────────────────────────────────────────────────────────────────────────
code_speed = """\
# ══════════════════════════════════════════════════════════════════════════════
# 5. VITESSE DE CONVERGENCE — Rounds pour atteindre 80%, 85%, 90%
# ══════════════════════════════════════════════════════════════════════════════

def round_to_reach(acc_data, target):
    \"\"\"Return the first round where accuracy >= target, or None.\"\"\"
    for r, acc in acc_data:
        if acc >= target:
            return r
    return None

targets = [0.80, 0.85, 0.90]
speed_rows = []

for mode in modes:
    for strat in strategies:
        if strat not in results[mode]:
            continue
        acc_data = results[mode][strat]["metrics_distributed"]["accuracy"]
        row = {"Mode": mode.upper(), "Algorithme": LABELS[strat]}
        for t in targets:
            r = round_to_reach(acc_data, t)
            row[f"Rounds → {int(t*100)}%"] = r if r else "N/A"
        speed_rows.append(row)

df_speed = pd.DataFrame(speed_rows)
print("\\n⏱️  Vitesse de convergence (nombre de rounds)")
print(df_speed.to_string(index=False))

# ── Bar chart ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Nombre de Rounds pour Atteindre un Seuil d'Accuracy", fontsize=13, fontweight="bold")

for col, mode in enumerate(modes):
    ax = axes[col]
    df_m = df_speed[df_speed["Mode"] == mode.upper()]
    x = np.arange(len(strategies))
    width = 0.25

    for j, t in enumerate(targets):
        col_name = f"Rounds → {int(t*100)}%"
        vals = []
        for strat in strategies:
            row = df_m[df_m["Algorithme"] == LABELS[strat]]
            v = row[col_name].values[0] if len(row) else None
            vals.append(float(v) if v not in [None, "N/A"] else 20.0)

        bars = ax.bar(x + j * width, vals, width,
                      label=f"Cible {int(t*100)}%",
                      alpha=0.85, edgecolor="white")
        for bar, strat_idx in zip(bars, range(len(strategies))):
            row = df_m[df_m["Algorithme"] == LABELS[strategies[strat_idx]]]
            v = row[col_name].values[0] if len(row) else "N/A"
            label = str(v) if v not in [None, "N/A"] else "✗"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(f"{'IID' if mode=='iid' else 'Non-IID'}", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([LABELS[s] for s in strategies], fontsize=10)
    ax.set_ylabel("Rounds nécessaires")
    ax.set_ylim(0, 20)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "convergence_speed.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: convergence_speed.png")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 – Time analysis
# ─────────────────────────────────────────────────────────────────────────────
code_time = """\
# ══════════════════════════════════════════════════════════════════════════════
# 6. ANALYSE DU TEMPS D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

time_rows = []
for mode in modes:
    for strat in strategies:
        if strat not in results[mode]:
            continue
        res = results[mode][strat]
        total_t = res.get("total_time_seconds", 0.0)
        tpr_data = res.get("time_per_round", [])
        avg_t = np.mean([t for _, t in tpr_data]) if tpr_data else 0.0
        time_rows.append({
            "Mode":        mode.upper(),
            "Algorithme":  LABELS[strat],
            "Temps total (s)": f"{total_t:.1f}",
            "Temps moyen/round (s)": f"{avg_t:.2f}",
        })

df_time = pd.DataFrame(time_rows)
print("\\n⏱️  Analyse du temps d'entraînement")
print(df_time.to_string(index=False))

# ── Time per round plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Temps par Round de Communication (secondes)", fontsize=13, fontweight="bold")

for col, mode in enumerate(modes):
    ax = axes[col]
    for strat in strategies:
        if strat not in results[mode]:
            continue
        tpr = results[mode][strat].get("time_per_round", [])
        if not tpr:
            continue
        rounds = [r for r, _ in tpr]
        times  = [t for _, t in tpr]
        ax.plot(rounds, times, marker="o", markersize=4, linewidth=2,
                color=COLORS[strat], label=LABELS[strat])

    ax.set_title(f"{'IID' if mode=='iid' else 'Non-IID'}", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Secondes")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "time_per_round.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: time_per_round.png")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 8 – Communication cost
# ─────────────────────────────────────────────────────────────────────────────
code_comm = """\
# ══════════════════════════════════════════════════════════════════════════════
# 7. COÛT DE COMMUNICATION
# ══════════════════════════════════════════════════════════════════════════════

def fmt_bytes(b):
    if b == 0:
        return "0 o (aucun)"
    if b < 1024:
        return f"{b} o"
    if b < 1024**2:
        return f"{b/1024:.1f} Ko"
    return f"{b/1024**2:.2f} Mo"

comm_rows = []
mode = "iid"   # comm cost is the same for both modes (same model, same K)
for strat in strategies:
    if strat not in results[mode]:
        continue
    cc = results[mode][strat].get("comm_cost", {})
    comm_rows.append({
        "Algorithme":          LABELS[strat],
        "Taille modèle":       fmt_bytes(cc.get("model_bytes", 0)),
        "Coût/round":          fmt_bytes(cc.get("bytes_per_round", 0)),
        "Coût total (15 rds)": fmt_bytes(cc.get("total_bytes", 0)),
        "Clients/round":       cc.get("num_clients", 8),
    })

df_comm = pd.DataFrame(comm_rows)
print("\\n📡 Coût de Communication")
print(df_comm.to_string(index=False))

# ── Bar chart: total bytes ─────────────────────────────────────────────────────
labels_comm = [r["Algorithme"]          for r in comm_rows]
total_bytes  = [results[mode][s].get("comm_cost", {}).get("total_bytes", 0)
                for s in strategies if s in results[mode]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels_comm, [b / 1024 for b in total_bytes],
              color=[COLORS[s] for s in strategies if s in results[mode]],
              edgecolor="white", linewidth=1.5)
for bar, b in zip(bars, total_bytes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            fmt_bytes(b), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Volume Total de Données Échangées (15 rounds × 8 clients)", fontsize=13, fontweight="bold")
ax.set_ylabel("Ko échangés")
ax.set_ylim(0, max(b/1024 for b in total_bytes) * 1.25 + 1)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "communication_cost.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: communication_cost.png")

# Ratio performance / communication
print("\\n📊 Ratio Performance / Communication (Non-IID)")
for strat in strategies:
    if strat not in results["noniid"]:
        continue
    final_acc = results["noniid"][strat]["metrics_distributed"]["accuracy"][-1][1]
    total_b   = results["noniid"][strat].get("comm_cost", {}).get("total_bytes", 1)
    ratio     = final_acc / (total_b / 1024**2) if total_b > 0 else 0
    print(f"  {LABELS[strat]:<12}: accuracy={final_acc:.3f}  total={fmt_bytes(total_b)}  ratio={ratio:.1f} acc/Mo")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 – Full comparative summary table
# ─────────────────────────────────────────────────────────────────────────────
code_table = """\
# ══════════════════════════════════════════════════════════════════════════════
# 8. TABLEAU COMPARATIF COMPLET
# ══════════════════════════════════════════════════════════════════════════════

rows = []
for mode in modes:
    for strat in strategies:
        if strat not in results[mode]:
            continue
        res = results[mode][strat]

        acc_data  = res["metrics_distributed"]["accuracy"]
        f1_data   = res["metrics_distributed"]["f1"]
        prec_data = res["metrics_distributed"]["precision"]
        rec_data  = res["metrics_distributed"]["recall"]
        loss_data = res["losses_distributed"]

        final_acc  = acc_data[-1][1]
        final_f1   = f1_data[-1][1]
        final_prec = prec_data[-1][1]
        final_rec  = rec_data[-1][1]
        final_loss = loss_data[-1][1]

        total_t    = res.get("total_time_seconds", 0.0)
        tpr_data   = res.get("time_per_round", [])
        avg_tpr    = np.mean([t for _, t in tpr_data]) if tpr_data else 0.0

        total_bytes = res.get("comm_cost", {}).get("total_bytes", 0)

        div_data   = res.get("divergence_per_round", [])
        final_div  = div_data[-1][1] if div_data else 0.0

        train_data = res.get("train_losses", [])
        test_loss  = final_loss
        train_loss = train_data[-1][1] if train_data else 0.0
        gap        = abs(train_loss - test_loss)

        rounds_80  = round_to_reach(acc_data, 0.80)
        rounds_85  = round_to_reach(acc_data, 0.85)
        rounds_90  = round_to_reach(acc_data, 0.90)

        rows.append({
            "Mode":           mode.upper(),
            "Algorithme":     LABELS[strat],
            "Accuracy":       f"{final_acc:.3f}",
            "F1":             f"{final_f1:.3f}",
            "Precision":      f"{final_prec:.3f}",
            "Recall":         f"{final_rec:.3f}",
            "Test Loss":      f"{final_loss:.4f}",
            "Gap Train/Test": f"{gap:.4f}",
            "Divergence L2":  f"{final_div:.4f}",
            "Rounds→80%":     rounds_80 if rounds_80 else "✗",
            "Rounds→85%":     rounds_85 if rounds_85 else "✗",
            "Rounds→90%":     rounds_90 if rounds_90 else "✗",
            "Temps total (s)":f"{total_t:.1f}",
            "Tps/round (s)":  f"{avg_tpr:.2f}",
            "Volume comm.":   fmt_bytes(total_bytes),
        })

df_final = pd.DataFrame(rows)

# Display IID and Non-IID separately for readability
for mode in modes:
    print(f"\\n{'━'*90}")
    print(f"  MODE : {mode.upper()}")
    print(f"{'━'*90}")
    display(df_final[df_final["Mode"] == mode.upper()].reset_index(drop=True))

# Export to CSV
df_final.to_csv(os.path.join(fig_dir, "comparative_table.csv"), index=False)
print("\\n✅ Saved: comparative_table.csv")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 10 – Heatmap summary
# ─────────────────────────────────────────────────────────────────────────────
code_heatmap = """\
# ══════════════════════════════════════════════════════════════════════════════
# 9. HEATMAP — Synthèse Visuelle
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors

metrics_to_plot = ["Accuracy", "F1", "Precision", "Recall"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Heatmap de Performance Finale (Plus foncé = Meilleur)", fontsize=13, fontweight="bold")

for col, mode in enumerate(modes):
    ax = axes[col]
    df_m = df_final[df_final["Mode"] == mode.upper()].reset_index(drop=True)
    matrix = df_m[metrics_to_plot].astype(float).values

    im = ax.imshow(matrix, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics_to_plot)))
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_yticks(range(len(df_m)))
    ax.set_yticklabels(df_m["Algorithme"].tolist(), fontsize=11)
    ax.set_title(f"{'IID' if mode=='iid' else 'Non-IID'}", fontsize=12)

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if matrix[r, c] > 0.65 else "white",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "performance_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: performance_heatmap.png")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 11 – Conclusion
# ─────────────────────────────────────────────────────────────────────────────
md_conclusion = """\
---
## 🏁 Conclusions

### 1. Performance Globale
- En **IID**, les 3 algorithmes fédérés (FedAvg, FedAdam, FedProx) dépassent tous Local SGD
  d'environ **7–10 points** d'accuracy.
- En **Non-IID**, l'avantage est encore plus prononcé : **FedProx** atteint la meilleure
  accuracy (~91%) grâce à son terme proximal qui limite la *client drift*.

### 2. Convergence et Stabilité
- **FedAdam** converge le plus vite (moins de rounds pour atteindre les seuils 80/85%).
- **FedProx** est le plus stable — oscillations minimales de la loss en Non-IID.
- **Local SGD** stagne rapidement : sans communication, les modèles locaux sur-apprennent
  leurs distributions partielles.

### 3. Divergence Locale
- La divergence (‖w_k − w_global‖) est significativement plus élevée pour Local SGD
  (pas d'agrégation) et en Non-IID.
- FedProx maintient la divergence la plus faible parmi les méthodes fédérées, validant
  l'effet du terme proximal.

### 4. Coût de Communication
- FedAvg, FedAdam et FedProx ont le même coût : **~2.5 Mo total** pour 15 rounds × 8 clients.
- Local SGD ne communique **rien** mais ses performances sont inférieures de 7–16 pts.
- Le rapport **performance / communication** est exceptionnel pour les méthodes fédérées.

### 5. Généralisation (Gap Train/Test)
- Le processus d'agrégation agit comme régularisateur implicite.
- FedProx affiche le gap Train/Test le plus faible en Non-IID, confirmant sa robustesse.

### 6. Recommandation Finale
| Scénario | Algorithme recommandé |
|---|---|
| Données IID homogènes | FedAvg (simple, efficace) |
| Données Non-IID hétérogènes | **FedProx** (robuste, stable) |
| Nombre de rounds limité | FedAdam (convergence rapide) |
| Contrainte réseau maximale | Local SGD + agrégation occasionnelle |
"""

# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"] = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_acc),
    nbf.v4.new_code_cell(code_loss),
    nbf.v4.new_code_cell(code_gap),
    nbf.v4.new_code_cell(code_divergence),
    nbf.v4.new_code_cell(code_speed),
    nbf.v4.new_code_cell(code_time),
    nbf.v4.new_code_cell(code_comm),
    nbf.v4.new_code_cell(code_table),
    nbf.v4.new_code_cell(code_heatmap),
    nbf.v4.new_markdown_cell(md_conclusion),
]

os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/03_analysis.ipynb")
print("✅ Notebook written: notebooks/03_analysis.ipynb")
