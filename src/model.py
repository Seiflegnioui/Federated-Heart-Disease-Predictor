import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_dim=23):
        super(HeartDiseaseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = nn.BCEWithLogitsLoss()
    net.train()
    net.to(device)
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = nn.BCEWithLogitsLoss()
    net.eval()
    net.to(device)
    
    loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    loss /= len(testloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return loss, accuracy, f1, precision, recall
