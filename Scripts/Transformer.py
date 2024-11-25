#Transformer.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# input and output paths
train_csv = "Data/Model/Train.csv"
val_csv = "Data/Model/Validation.csv"
test_csv = "Data/Model/Test.csv"
embeddings_116_dir = "Data/embeddings/116-bills"
embeddings_117_dir = "Data/embeddings/117-bills"

model_dir = "Data/Model/Transformer"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "TransformerModel.pth")
report_path = os.path.join(model_dir, "TestReport.txt")

# mps for faster training
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# dataset labeling
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
        embedding = np.load(embedding_path).astype(np.float32)  
        return torch.tensor(embedding), torch.tensor(label)

# load the splits
train_dataset = EmbeddingDataset(train_csv, embeddings_116_dir, embeddings_117_dir)
val_dataset = EmbeddingDataset(val_csv, embeddings_116_dir, embeddings_117_dir)
test_dataset = EmbeddingDataset(test_csv, embeddings_116_dir, embeddings_117_dir)

# again using larger batch size
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# emplyoing classic transformer model with 2 heads and 2 layers
# chunking the embedding into 16 "tokens"
# an incorporating a positional encoding to help the model learn the order of the chunks
# which should allow for not having to renormalize each chunk of the embedding vector.
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_chunks, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.num_chunks = num_chunks
        self.chunk_size = input_dim // num_chunks

        # positional encoding inserted -- pulling from llm architecture in using learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(num_chunks, d_model))

        # .3 dropout and relu activation for better learning and gradient flow
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.3,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 21 headed classifier for each committee
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_chunks * d_model, num_classes)
        )
    # this is to help the model learn the order of the chunks, see above
    def forward(self, x):
        x = x.view(-1, self.num_chunks, self.chunk_size)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        return self.classifier(x)

# transformer hyperparameters
input_dim = 3072
num_chunks = 16  
d_model = 192    
nhead = 2 # attention heads
num_layers = 2   
num_classes = 21  # number of committees for output classification head
model = TransformerClassifier(input_dim, num_chunks, d_model, nhead, num_layers, num_classes).to(device)

# standard loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

        # val
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

# train for 50 epochs!
train_model(model, train_loader, val_loader, num_epochs=50)

# Ssave mode
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# test it
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