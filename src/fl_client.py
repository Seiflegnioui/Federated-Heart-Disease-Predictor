import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from collections import OrderedDict
from model import HeartDiseaseMLP, train, test

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, cid, data_dir, model, device):
        self.cid = cid
        self.device = device
        self.model = model
        
        # Load local data
        X_path = os.path.join(data_dir, f"client_{cid}_X.npy")
        y_path = os.path.join(data_dir, f"client_{cid}_y.npy")
        
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)
        
        # Split 80/20 train/test
        n_train = int(len(X) * 0.8)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        
        self.trainloader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=32, shuffle=True
        )
        self.testloader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=32, shuffle=False
        )
        self.num_examples = {"trainset": len(X_train), "testset": len(X_test)}

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))
        epochs = config.get("epochs", 1)
        
        train(self.model, self.trainloader, optimizer, epochs, self.device)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1, precision, recall = test(self.model, self.testloader, self.device)
        return loss, self.num_examples["testset"], {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
