import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import copy
from collections import OrderedDict
from model import HeartDiseaseMLP, test
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ── Communication cost helper ──────────────────────────────────────────────────

def compute_model_size_bytes(model: nn.Module) -> int:
    """Return the total size in bytes of all model parameters (float32 = 4 bytes)."""
    return sum(p.numel() * 4 for p in model.parameters())


# ── Local client ───────────────────────────────────────────────────────────────

class LocalClient:
    def __init__(self, cid, data_dir, device):
        self.cid = cid
        self.device = device

        X_path = os.path.join(data_dir, f"client_{cid}_X.npy")
        y_path = os.path.join(data_dir, f"client_{cid}_y.npy")
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)

        n_train = int(len(X) * 0.8)
        self.X_train, self.y_train = X[:n_train], y[:n_train]
        self.X_test,  self.y_test  = X[n_train:], y[n_train:]

        self.trainloader = DataLoader(
            TensorDataset(torch.tensor(self.X_train), torch.tensor(self.y_train)),
            batch_size=32, shuffle=True
        )
        self.testloader = DataLoader(
            TensorDataset(torch.tensor(self.X_test), torch.tensor(self.y_test)),
            batch_size=32, shuffle=False
        )
        self.num_train = len(self.X_train)
        self.num_test  = len(self.X_test)

    # ------------------------------------------------------------------
    def train(self, global_model, epochs, lr, strategy="FedAvg", mu=0.1):
        """Train locally and return (updated_weights, num_samples, train_loss)."""
        model = copy.deepcopy(global_model)
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        global_params = [p.clone().detach() for p in global_model.parameters()]

        epoch_loss = 0.0
        n_batches  = 0
        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if strategy == "FedProx":
                    proximal_term = 0.0
                    for param, global_param in zip(model.parameters(), global_params):
                        proximal_term += ((param - global_param) ** 2).sum()
                    loss += (mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        updated_weights = [p.cpu().detach().numpy() for p in model.parameters()]
        return updated_weights, self.num_train, avg_train_loss

    # ------------------------------------------------------------------
    def evaluate(self, model):
        loss, accuracy, f1, precision, recall = test(model, self.testloader, self.device)
        return loss, self.num_test, accuracy, f1, precision, recall


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(weights_results):
    """Weighted average of model weights (FedAvg aggregation)."""
    total_samples = sum(n for w, n in weights_results)
    agg_weights = []
    for i, (weights, n) in enumerate(weights_results):
        frac = n / total_samples
        if i == 0:
            agg_weights = [w * frac for w in weights]
        else:
            agg_weights = [aw + w * frac for aw, w in zip(agg_weights, weights)]
    return agg_weights


# ── Global evaluation ─────────────────────────────────────────────────────────

def evaluate_global(global_model, clients):
    total_loss, total_acc, total_f1, total_prec, total_rec = 0, 0, 0, 0, 0
    total_samples = 0
    for client in clients:
        loss, n, acc, f1, prec, rec = client.evaluate(global_model)
        total_loss  += loss * n
        total_acc   += acc  * n
        total_f1    += f1   * n
        total_prec  += prec * n
        total_rec   += rec  * n
        total_samples += n

    return {
        "loss":      total_loss  / total_samples,
        "accuracy":  total_acc   / total_samples,
        "f1":        total_f1    / total_samples,
        "precision": total_prec  / total_samples,
        "recall":    total_rec   / total_samples,
    }


# ── Local divergence helper ───────────────────────────────────────────────────

def compute_divergence(local_weights_list, global_model):
    """
    Mean L2 norm of (local_weights - global_weights) across all clients.
    Measures how much local models drift from the global model.
    """
    global_params = [p.detach().numpy() for p in global_model.parameters()]
    divergences = []
    for local_weights in local_weights_list:
        diff = sum(
            np.linalg.norm(lw - gw) ** 2
            for lw, gw in zip(local_weights, global_params)
        )
        divergences.append(np.sqrt(diff))
    return float(np.mean(divergences))


# ── Main federated simulation ─────────────────────────────────────────────────

def run_simulation(strategy_name: str, data_mode: str, num_rounds: int, epochs: int = 5):
    print(f"\n{'='*55}")
    print(f"  {strategy_name}  |  {data_mode.upper()}  |  {num_rounds} rounds")
    print(f"{'='*55}\n")

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "outputs", f"clients_{data_mode}"
    )
    device  = torch.device("cpu")
    clients = [LocalClient(cid, data_dir, device) for cid in range(8)]
    global_model = HeartDiseaseMLP().to(device)

    model_bytes = compute_model_size_bytes(global_model)

    # FedAdam server-side state
    if strategy_name == "FedAdam":
        m_t = [np.zeros_like(p.detach().numpy()) for p in global_model.parameters()]
        v_t = [np.zeros_like(p.detach().numpy()) for p in global_model.parameters()]
        beta1, beta2, eta, tau = 0.9, 0.99, 0.01, 1e-3

    history = {
        "losses_distributed":  [],
        "train_losses":        [],           # NEW – train loss per round
        "metrics_distributed": {
            "accuracy": [], "f1": [], "precision": [], "recall": []
        },
        "time_per_round":      [],           # NEW – seconds per round
        "total_time_seconds":  0.0,          # NEW – cumulative training time
        "divergence_per_round": [],          # NEW – mean L2 divergence per round
        "comm_cost": {                        # NEW – communication cost
            "model_bytes":       model_bytes,
            "bytes_per_round":   model_bytes * 2 * 8,   # upload + download × 8 clients
            "total_bytes":       model_bytes * 2 * 8 * num_rounds,
            "num_rounds":        num_rounds,
            "num_clients":       8,
        },
    }

    total_start = time.perf_counter()

    for r in range(1, num_rounds + 1):
        round_start = time.perf_counter()

        weights_results = []
        local_weights_only = []
        round_train_loss = 0.0
        total_train_samples = 0

        for client in clients:
            weights, n, train_loss = client.train(
                global_model, epochs=epochs, lr=0.001, strategy=strategy_name
            )
            weights_results.append((weights, n))
            local_weights_only.append(weights)
            round_train_loss    += train_loss * n
            total_train_samples += n

        agg_weights = aggregate(weights_results)

        # ── Divergence (before updating global model) ──────────────────────
        divergence = compute_divergence(local_weights_only, global_model)
        history["divergence_per_round"].append((r, divergence))

        # ── FedAdam server update ──────────────────────────────────────────
        if strategy_name == "FedAdam":
            global_weights = [p.detach().numpy() for p in global_model.parameters()]
            delta_t = [gw - aw for gw, aw in zip(global_weights, agg_weights)]

            for i in range(len(m_t)):
                m_t[i] = beta1 * m_t[i] + (1 - beta1) * delta_t[i]
                v_t[i] = beta2 * v_t[i] + (1 - beta2) * (delta_t[i] ** 2)
                global_weights[i] = global_weights[i] - eta * m_t[i] / (np.sqrt(v_t[i]) + tau)
            new_params = global_weights
        else:
            new_params = agg_weights

        # ── Update global model ────────────────────────────────────────────
        params_dict = zip(global_model.state_dict().keys(), new_params)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)

        # ── Evaluate & record ──────────────────────────────────────────────
        metrics = evaluate_global(global_model, clients)

        round_elapsed = time.perf_counter() - round_start
        avg_train_loss = round_train_loss / max(total_train_samples, 1)

        history["losses_distributed"].append((r, metrics["loss"]))
        history["train_losses"].append((r, avg_train_loss))
        history["metrics_distributed"]["accuracy"].append((r, metrics["accuracy"]))
        history["metrics_distributed"]["f1"].append((r, metrics["f1"]))
        history["metrics_distributed"]["precision"].append((r, metrics["precision"]))
        history["metrics_distributed"]["recall"].append((r, metrics["recall"]))
        history["time_per_round"].append((r, round_elapsed))

        print(
            f"  Round {r:>2}: loss={metrics['loss']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"divergence={divergence:.4f}  "
            f"time={round_elapsed:.2f}s"
        )

    history["total_time_seconds"] = time.perf_counter() - total_start
    print(f"\n  Total training time : {history['total_time_seconds']:.1f}s")

    # ── Save results ───────────────────────────────────────────────────────
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{strategy_name}_{data_mode}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save best model for FedProx Non-IID (reference model)
    if strategy_name == "FedProx" and data_mode == "noniid":
        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "models"
        )
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            global_model.state_dict(),
            os.path.join(model_dir, "global_model_fedprox_noniid.pth")
        )
        print("  [OK] Saved global model -> outputs/models/global_model_fedprox_noniid.pth")


# ── Local SGD baseline ────────────────────────────────────────────────────────

def run_local_sgd(data_mode: str, num_rounds: int, epochs: int = 5):
    print(f"\n{'='*55}")
    print(f"  Local SGD  |  {data_mode.upper()}  |  {num_rounds} rounds")
    print(f"{'='*55}\n")

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "outputs", f"clients_{data_mode}"
    )
    device  = torch.device("cpu")
    clients = [LocalClient(cid, data_dir, device) for cid in range(8)]

    local_models = [HeartDiseaseMLP().to(device) for _ in range(8)]
    optimizers   = [torch.optim.Adam(m.parameters(), lr=0.001) for m in local_models]
    criterion    = nn.BCEWithLogitsLoss()

    dummy_global = HeartDiseaseMLP().to(device)   # fixed reference for divergence
    model_bytes  = compute_model_size_bytes(dummy_global)

    history = {
        "losses_distributed":   [],
        "train_losses":         [],
        "metrics_distributed":  {
            "accuracy": [], "f1": [], "precision": [], "recall": []
        },
        "time_per_round":       [],
        "total_time_seconds":   0.0,
        "divergence_per_round": [],    # divergence vs shared initial weights
        "comm_cost": {
            "model_bytes":      model_bytes,
            "bytes_per_round":  0,     # No communication in Local SGD
            "total_bytes":      0,
            "num_rounds":       num_rounds,
            "num_clients":      8,
        },
    }

    total_start = time.perf_counter()
    # Record initial global weights as reference for divergence calculation
    ref_params = [p.detach().numpy() for p in dummy_global.parameters()]

    for r in range(1, num_rounds + 1):
        round_start = time.perf_counter()

        round_train_loss    = 0.0
        total_train_samples = 0

        # ── Train locally ──────────────────────────────────────────────────
        for i, client in enumerate(clients):
            model = local_models[i]
            opt   = optimizers[i]
            model.train()
            batch_loss = 0.0
            n_batches  = 0
            for ep in range(epochs):
                for inputs, labels in client.trainloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float().unsqueeze(1)
                    opt.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    opt.step()
                    batch_loss += loss.item()
                    n_batches  += 1
            round_train_loss    += (batch_loss / max(n_batches, 1)) * client.num_train
            total_train_samples += client.num_train

        # ── Divergence (local models vs fixed reference) ───────────────────
        local_weights = [
            [p.detach().numpy() for p in local_models[i].parameters()]
            for i in range(8)
        ]
        divergences = []
        for lw in local_weights:
            diff = sum(np.linalg.norm(l - g) ** 2 for l, g in zip(lw, ref_params))
            divergences.append(np.sqrt(diff))
        divergence = float(np.mean(divergences))
        history["divergence_per_round"].append((r, divergence))

        # ── Evaluate ───────────────────────────────────────────────────────
        total_loss, total_acc = 0.0, 0.0
        total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
        total_samples = 0
        for i, client in enumerate(clients):
            loss, n, acc, f1, prec, rec = client.evaluate(local_models[i])
            total_loss  += loss * n
            total_acc   += acc  * n
            total_f1    += f1   * n
            total_prec  += prec * n
            total_rec   += rec  * n
            total_samples += n

        round_elapsed  = time.perf_counter() - round_start
        avg_train_loss = round_train_loss / max(total_train_samples, 1)

        history["losses_distributed"].append((r, total_loss  / total_samples))
        history["train_losses"].append((r, avg_train_loss))
        history["metrics_distributed"]["accuracy"].append((r, total_acc  / total_samples))
        history["metrics_distributed"]["f1"].append((r, total_f1   / total_samples))
        history["metrics_distributed"]["precision"].append((r, total_prec / total_samples))
        history["metrics_distributed"]["recall"].append((r, total_rec  / total_samples))
        history["time_per_round"].append((r, round_elapsed))

        print(
            f"  Round {r:>2}: loss={total_loss/total_samples:.4f}  "
            f"acc={total_acc/total_samples:.4f}  "
            f"divergence={divergence:.4f}  "
            f"time={round_elapsed:.2f}s"
        )

    history["total_time_seconds"] = time.perf_counter() - total_start
    print(f"\n  Total training time : {history['total_time_seconds']:.1f}s")

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"LocalSGD_{data_mode}.json"), "w") as f:
        json.dump(history, f, indent=2)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    for mode in ["iid", "noniid"]:
        run_local_sgd(mode, num_rounds=args.rounds, epochs=args.epochs)
        for strategy in ["FedAvg", "FedAdam", "FedProx"]:
            run_simulation(strategy, mode, num_rounds=args.rounds, epochs=args.epochs)
