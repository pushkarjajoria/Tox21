from torch.utils.data import DataLoader
# from data_prep import FingerprintDataset
from data_prep_transformer import SmilesDataset
from eval import get_all_pred_and_labels
from tqdm import tqdm
import torch
import numpy as np
from model import MolPropPredictor, NoiseLayer, MolPropPredictorMolFormer
from utils import hybrid_train, test, NoisedDataset, calculate_positive_percentage

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

# Dataset preparation
# train_dataset = FingerprintDataset('./benchmark_datasets/tox21/train.smi')
# test_dataset = FingerprintDataset('./benchmark_datasets/tox21/test.smi')

train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi')
test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi')

# Using the noised dataset for training
NOISE_LEVEL = 0.0
train_dataset = NoisedDataset(original_dataset=train_dataset, noise_level=NOISE_LEVEL)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)  # No shuffle for evaluation


train_positive_percentage = calculate_positive_percentage(train_dataset)
test_positive_percentage = calculate_positive_percentage(test_dataset)
print(f'Percentage of positive test results in training dataset: {train_positive_percentage:.2f}%')
print(f'Percentage of positive test results in testing dataset: {test_positive_percentage:.2f}%')


# Model setup
# Calculate weights for both classes
positive_weight = (100 - train_positive_percentage) / 100.0
negative_weight = train_positive_percentage / 100.0
class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32).to(device)  # Weight for both classes

inp_size = len(train_dataset.x)
baseline_model = MolPropPredictorMolFormer().to(device)
optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)  # Use CrossEntropyLoss

epochs = 50

# Training loop
for epoch in tqdm(range(epochs)):
    baseline_model.train()
    for batch in train_data_loader:
        x = batch['x']
        labels = batch['label'].long().to(device)  # Labels should be of type long for CrossEntropyLoss
        optim.zero_grad()
        output = baseline_model(x)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

test(test_data_loader, baseline_model)

baseline_output, y_train_noise = get_all_pred_and_labels(baseline_model, train_data_loader)
baseline_confusion = np.zeros((2, 2))
for n, p in zip(y_train_noise, baseline_output):
    n = n.cpu().numpy()
    p = p.cpu().numpy()
    baseline_confusion[p, n] += 1.

channel_weights = baseline_confusion.copy()
channel_weights /= channel_weights.sum(axis=1, keepdims=True)
channel_weights = np.log(channel_weights + 1e-8)
channel_weights = torch.from_numpy(channel_weights)
channel_weights = channel_weights.float()
noisemodel = NoiseLayer(theta=channel_weights.to(device), k=2)
noise_optimizer = torch.optim.Adam(noisemodel.parameters(),
                             lr=1e-3)

print("noisy channel finished.")
# noisy model train and test
for epoch in tqdm(range(epochs)):
    hybrid_train(train_data_loader, baseline_model, noisemodel, optim, noise_optimizer, criterion)

print("After hybrid, test acc: ")
test(test_data_loader, baseline_model)
print("Finished hybrid training.")

# Evaluation after training
