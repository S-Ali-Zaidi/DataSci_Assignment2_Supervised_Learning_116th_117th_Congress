#FCNN.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib #ignre, not used in this script

# Paths
train_csv = "Data/Model/Train.csv"
val_csv = "Data/Model/Validation.csv"
test_csv = "Data/Model/Test.csv"
embeddings_116_dir = "Data/embeddings/116-bills"
embeddings_117_dir = "Data/embeddings/117-bills"

model_dir = "Data/Model/FCNN"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "FCNNModel.pth")
report_path = os.path.join(model_dir, "TestReport.txt")

# Use MPS to train faster on Mac's mps GPU backend
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class EmbeddingDataset(Dataset):
    def __init__(self, csv_path, embeddings_116_dir, embeddings_117_dir):
        self.data = pd.read_csv(csv_path)
        self.embeddings_116_dir = embeddings_116_dir
        self.embeddings_117_dir = embeddings_117_dir
        self.label_encoder = LabelEncoder()
        self.data["Committee Name"] = self.label_encoder.fit_transform(self.data["Committee Name"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        congress = row["Congress"]
        bill_name = row["Bill Name"]
        label = row["Committee Name"]
        embeddings_dir = self.embeddings_116_dir if congress == 116 else self.embeddings_117_dir
        embedding_path = os.path.join(embeddings_dir, f"{bill_name}.npy")
        embedding = np.load(embedding_path).astype(np.float32)  # Ensure float32 for PyTorch
        return torch.tensor(embedding), torch.tensor(label)

# load data
train_dataset = EmbeddingDataset(train_csv, embeddings_116_dir, embeddings_117_dir)
val_dataset = EmbeddingDataset(val_csv, embeddings_116_dir, embeddings_117_dir)
test_dataset = EmbeddingDataset(test_csv, embeddings_116_dir, embeddings_117_dir)

# set data loaders -- using a larggee btch size to speed up training
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# FCNN model architecture -- i scaled up the typical mnist style FCNN to handle the larger embeddings
# using dropout for regularization 
class CommitteeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(CommitteeClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.network(x)

# initialize the model
input_dim = 3072
hidden_dims = [512, 128]
output_dim = 21  # Number of committees
model = CommitteeClassifier(input_dim, hidden_dims, output_dim).to(device)

# standard adam optimizer and cross entropy loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 50 epoch training loop with validation
def train_model(model, train_loader, val_loader, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {correct/total:.4f}")

# trin!!!
train_model(model, train_loader, val_loader, num_epochs=50)

# save model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# test against test split
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # save the report
    report = classification_report(
        true_labels,
        predictions,
        target_names=train_dataset.label_encoder.classes_,
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Test report saved to {report_path}")

    print("\n=== Test Performance ===")
    print(report)

evaluate_model(model, test_loader)