import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_prep import FingerprintDataset
from data_prep_transformer import SmilesDataset
from model import Tox21Predictor
from utils import test, calculate_positive_percentage, \
    EarlyStopping, validation_loss_tox21, \
    iterative_validation_split

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

patience = 3

# Dataset preparation
train_file = '/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/tox21/tox21_10k_data_all_duplicates_merged.csv'
train_dataset = FingerprintDataset(train_file)
test_file = '/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/tox21/tox21_10k_challenge_test_duplicates_merged.csv'
test_dataset = FingerprintDataset(test_file)
fingerprint = False if type(train_dataset) == SmilesDataset else True

# Input about PCA Transformation
pca_transform_data = False
n_pca_components = 256

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

train_data_loader, valid_data_loader, pca = iterative_validation_split(train_dataset, pca_transformation=pca_transform_data,
                                                                  n_pca_components=n_pca_components)
if pca_transform_data:
    test_dataset.fit_transform_pca(pca=pca)
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)  # No shuffle for evaluation

train_positive_percentage = calculate_positive_percentage(train_dataset)
test_positive_percentage = calculate_positive_percentage(test_dataset)
print(f'Percentage of positive test results in training dataset: {train_positive_percentage}%')
print(f'Percentage of positive test results in testing dataset: {test_positive_percentage}%')

# Model setup
# Calculate weights for both classes
positive_weight = (100 - train_positive_percentage) / 100.0
negative_weight = train_positive_percentage / 100.0
class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32).to(device).T  # Weight for both classes
class_weights = class_weights.mean(dim=0)
if pca_transform_data:
    input_size = n_pca_components
else:
    input_size = 2048
baseline_model = Tox21Predictor(input_size=input_size, output_size=12*2, seed=42).to(device)
optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights).to(device)

epochs = 50
early_stopping = EarlyStopping(patience=patience)
# Training loop
for epoch in tqdm(range(epochs)):
    baseline_model.train()
    running_loss = 0
    for batch in train_data_loader:
        if fingerprint:
            x = batch['x'].float().to(device) # Fingerprint Input
        else:
            x = batch['x']  # Smile input
        labels = batch['label'].nan_to_num().long().to(device)  # Labels should be of type long for CrossEntropyLoss
        batch_len = x.shape[0]
        num_tasks = labels.shape[1]
        mask = batch['mask'].long().to(device)
        optim.zero_grad()
        output = baseline_model(x)
        loss = criterion(output.reshape((batch_len*num_tasks, -1)), labels.reshape((batch_len*num_tasks)))
        masked_loss = loss * mask.reshape(-1)
        final_loss = masked_loss.sum() / mask.sum()
        final_loss.backward()
        optim.step()
        running_loss += final_loss.item()

    av_val_loss = validation_loss_tox21(baseline_model, valid_data_loader, criterion, device, fingerprint=fingerprint)
    print(f'Epoch: {epoch + 1}/{epochs}, '
          f'Training Loss: {running_loss/len(train_data_loader):.4f}, '
          f'Validation Loss: {av_val_loss:.4f}, ')

    # Check early stopping
    early_stopping(av_val_loss, model=baseline_model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.", flush=True)
        early_stopping.load_best_model(baseline_model)
        break

print("\n\n")
print("Test set results:", flush=True)
bl_accuracy, bl_precision, bl_recall, bl_f1, roc_auc = test(test_data_loader, baseline_model, fingerprint=fingerprint, verbose=True)
print("-"*71 + "\n\n")
print("Train set results:")
bl_accuracy, bl_precision, bl_recall, bl_f1, roc_auc = test(train_data_loader, baseline_model, fingerprint=fingerprint, verbose=True)
print("-"*71 + "\n\n")
print("Validation set results:")
bl_accuracy, bl_precision, bl_recall, bl_f1, roc_auc = test(valid_data_loader, baseline_model, fingerprint=fingerprint, verbose=True)
print("-"*71)
