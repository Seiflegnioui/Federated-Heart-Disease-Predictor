import os
import json
import torch
import torch.nn as nn
import numpy as np
import copy
from collections import OrderedDict
from model import HeartDiseaseMLP, test
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
        self.X_test, self.y_test = X[n_train:], y[n_train:]
        
        self.trainloader = DataLoader(
            TensorDataset(torch.tensor(self.X_train), torch.tensor(self.y_train)),
            batch_size=32, shuffle=True
        )
        self.testloader = DataLoader(
            TensorDataset(torch.tensor(self.X_test), torch.tensor(self.y_test)),
            batch_size=32, shuffle=False
        )
        self.num_train = len(self.X_train)
        self.num_test = len(self.X_test)

    def train(self, global_model, epochs, lr, strategy="FedAvg", mu=0.1):
        model = copy.deepcopy(global_model)
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        global_params = [p.clone().detach() for p in global_model.parameters()]
        
        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)
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
                
        return [p.cpu().detach().numpy() for p in model.parameters()], self.num_train

    def evaluate(self, model):
        loss, accuracy, f1, precision, recall = test(model, self.testloader, self.device)
        return loss, self.num_test, accuracy, f1, precision, recall

def aggregate(weights_results):
    total_samples = sum(n for w, n in weights_results)
    agg_weights = []
    for i, (weights, n) in enumerate(weights_results):
        frac = n / total_samples
        if i == 0:
            agg_weights = [w * frac for w in weights]
        else:
            agg_weights = [aw + w * frac for aw, w in zip(agg_weights, weights)]
    return agg_weights

def evaluate_global(global_model, clients):
    total_loss, total_acc, total_f1, total_prec, total_rec = 0, 0, 0, 0, 0
    total_samples = 0
    for client in clients:
        loss, n, acc, f1, prec, rec = client.evaluate(global_model)
        total_loss += loss * n
        total_acc += acc * n
        total_f1 += f1 * n
        total_prec += prec * n
        total_rec += rec * n
        total_samples += n
        
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_acc / total_samples,
        "f1": total_f1 / total_samples,
        "precision": total_prec / total_samples,
        "recall": total_rec / total_samples
    }

def run_simulation(strategy_name: str, data_mode: str, num_rounds: int, epochs: int = 5):
    print(f"\n{'='*50}")
    print(f"--- Running {strategy_name} on {data_mode} data ---")
    print(f"{'='*50}\n")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", f"clients_{data_mode}")
    device = torch.device("cpu")
    
    clients = [LocalClient(cid, data_dir, device) for cid in range(8)]
    global_model = HeartDiseaseMLP().to(device)
    
    # Server-side optimizer state for FedAdam
    if strategy_name == "FedAdam":
        m_t = [np.zeros_like(p.detach().numpy()) for p in global_model.parameters()]
        v_t = [np.zeros_like(p.detach().numpy()) for p in global_model.parameters()]
        beta1, beta2, eta, tau = 0.9, 0.99, 0.01, 1e-3
    
    history = {"losses_distributed": [], "metrics_distributed": {"accuracy": [], "f1": [], "precision": [], "recall": []}}
    
    for r in range(1, num_rounds + 1):
        weights_results = []
        for client in clients:
            weights, n = client.train(global_model, epochs=epochs, lr=0.001, strategy=strategy_name)
            weights_results.append((weights, n))
            
        agg_weights = aggregate(weights_results)
        
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
            
        # Update global model
        params_dict = zip(global_model.state_dict().keys(), new_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        metrics = evaluate_global(global_model, clients)
        history["losses_distributed"].append((r, metrics["loss"]))
        history["metrics_distributed"]["accuracy"].append((r, metrics["accuracy"]))
        history["metrics_distributed"]["f1"].append((r, metrics["f1"]))
        history["metrics_distributed"]["precision"].append((r, metrics["precision"]))
        history["metrics_distributed"]["recall"].append((r, metrics["recall"]))
        
        print(f"Round {r}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{strategy_name}_{data_mode}.json"), "w") as f:
        json.dump(history, f)
        
    if strategy_name == "FedProx" and data_mode == "noniid":
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(model_dir, "global_model_fedprox_noniid.pth"))
        print(f"  [OK] Saved global model to outputs/models/global_model_fedprox_noniid.pth")
        
def run_local_sgd(data_mode: str, num_rounds: int, epochs: int = 5):
    print(f"\n{'='*50}")
    print(f"--- Running Local SGD on {data_mode} data ---")
    print(f"{'='*50}\n")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", f"clients_{data_mode}")
    device = torch.device("cpu")
    clients = [LocalClient(cid, data_dir, device) for cid in range(8)]
    
    # Each client keeps its own model
    local_models = [HeartDiseaseMLP().to(device) for _ in range(8)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.001) for m in local_models]
    criterion = nn.BCEWithLogitsLoss()
    
    history = {"losses_distributed": [], "metrics_distributed": {"accuracy": [], "f1": [], "precision": [], "recall": []}}
    
    for r in range(1, num_rounds + 1):
        # Train locally
        for i, client in enumerate(clients):
            model = local_models[i]
            opt = optimizers[i]
            model.train()
            for ep in range(epochs):
                for inputs, labels in client.trainloader:
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    opt.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    opt.step()
                    
        # Evaluate local models on their respective test sets
        total_loss, total_acc, total_f1, total_prec, total_rec = 0, 0, 0, 0, 0
        total_samples = 0
        for i, client in enumerate(clients):
            loss, n, acc, f1, prec, rec = client.evaluate(local_models[i])
            total_loss += loss * n
            total_acc += acc * n
            total_f1 += f1 * n
            total_prec += prec * n
            total_rec += rec * n
            total_samples += n
            
        history["losses_distributed"].append((r, total_loss / total_samples))
        history["metrics_distributed"]["accuracy"].append((r, total_acc / total_samples))
        history["metrics_distributed"]["f1"].append((r, total_f1 / total_samples))
        history["metrics_distributed"]["precision"].append((r, total_prec / total_samples))
        history["metrics_distributed"]["recall"].append((r, total_rec / total_samples))
        
        print(f"Round {r}: loss={total_loss / total_samples:.4f}, acc={total_acc / total_samples:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"LocalSGD_{data_mode}.json"), "w") as f:
        json.dump(history, f)

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
