import pickle
from torch.utils.data import DataLoader

from data_prep import FingerprintDataset, calculate_positive_percentage, NoisedDataset
from model import MolPropPredictor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np

CLASSIFICATION_THRESHOLD = 0.5

# Load tokens
with open("./benchmark_datasets/tox21/tokens.txt", 'rb') as f:
    tokens = pickle.load(f)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

# Dataset preparation
train_dataset = FingerprintDataset('./benchmark_datasets/tox21/train.smi')
test_dataset = FingerprintDataset('./benchmark_datasets/tox21/test.smi')

# Using the noised dataset for training
train_dataset = NoisedDataset(original_dataset=train_dataset, noise_level=0.3)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)  # No shuffle for evaluation


train_positive_percentage = calculate_positive_percentage(train_dataset)
test_positive_percentage = calculate_positive_percentage(test_dataset)
print(f'Percentage of positive test results in training dataset: {train_positive_percentage:.2f}%')
print(f'Percentage of positive test results in testing dataset: {test_positive_percentage:.2f}%')


# Model setup
class_weights = torch.tensor([(100-train_positive_percentage)/train_positive_percentage])
inp_size = train_dataset.fingerprints.shape[1]
assert train_dataset.fingerprints.shape[1] == test_dataset.fingerprints.shape[1]
model = MolPropPredictor(inp_size).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss(weight=class_weights).to(device)

epochs = 100

# Training loop
for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_data_loader:
        x = batch['fingerprint'].float().to(device)
        labels = batch['label'].float().to(device)
        optim.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optim.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')


# Evaluation after training
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_data_loader:
        x = batch['fingerprint'].float().to(device)
        labels = batch['label'].float().to(device)
        output = model(x)
        preds = output.cpu().numpy() > CLASSIFICATION_THRESHOLD  # Apply threshold for binary classification
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
