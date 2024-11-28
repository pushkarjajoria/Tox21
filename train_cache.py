import random
from datetime import datetime
import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_prep import create_train_test_val_splits
from data_prep_transformer import SmilesDataset
from eval import get_all_pred_and_labels_cache5
from model import HybridModel, Cache5AntagonistPredictor, Channel2D
from utils import calculate_positive_percentage, EarlyStopping, test_cache5, validation_loss_cache5, hybrid_train_mnist

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

patience = 5
BETA = 0.0
batch_size = 64
val_batch_size = 1000
seed = 42

# Dataset preparation
data_path = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
train_dataset, val_dataset, test_dataset = create_train_test_val_splits(data_path)

fingerprint = False if type(train_dataset) == SmilesDataset else True

# Input about PCA Transformation
pca_transform_data = True
n_pca_components = 1024

# Using the noised dataset for training

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


if pca_transform_data:
    pca = train_dataset.fit_transform_pca(n_components=n_pca_components)
# Create val dataset
if pca_transform_data:
    _ = val_dataset.fit_transform_pca(pca=pca)
# Create data loaders
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)

if pca_transform_data:
    test_dataset.fit_transform_pca(pca=pca)

test_data_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

train_positive_percentage = calculate_positive_percentage(train_dataset)
test_positive_percentage = calculate_positive_percentage(test_dataset)
print(f"Len of Train: {len(train_dataset)}")
print(f"Len of Test: {len(test_dataset)}")
print(f'Percentage of positive test results in training dataset: {train_positive_percentage}%')
print(f'Percentage of positive test results in testing dataset: {test_positive_percentage}%')

# Model setup
# Calculate weights for both classes
positive_weight = (100 - train_positive_percentage) / 100.0
negative_weight = train_positive_percentage / 100.0
class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32).to(device).T  # Weight for both classes
if pca_transform_data:
    input_size = n_pca_components
else:
    input_size = 2048
baseline_model = Cache5AntagonistPredictor(input_size=input_size, output_size=2, seed=seed).to(device)
optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=class_weights).to(device)

epochs = 50
early_stopping = EarlyStopping(patience=patience)
# Training loop
for epoch in tqdm(range(epochs)):
    baseline_model.train()
    running_loss = 0
    for batch in train_data_loader:
        if fingerprint:
            x = batch['x'].float().to(device)   # Fingerprint Input
        else:
            x = batch['x']  # Smile input
        labels = batch['label'].long().to(device)  # Labels should be of type long for CrossEntropyLoss
        batch_len = x.shape[0]
        # num_tasks = labels.shape[1]
        optim.zero_grad()
        output = baseline_model(x)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
        running_loss += loss.item()

    av_val_loss = validation_loss_cache5(baseline_model, val_data_loader, criterion, device, fingerprint=fingerprint)
    print(f'Epoch: {epoch + 1}/{epochs}, '
          f'Training Loss: {running_loss/len(train_data_loader):.4f}, '
          f'Validation Loss: {av_val_loss:.4f}, ')

    # Check early stopping
    early_stopping(av_val_loss, model=baseline_model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.", flush=True)
        early_stopping.load_best_model(baseline_model)
        break

accuracy_task_map, precision_task_map, recall_task_map, f1_task_map, roc_auc_task_map \
    = test_cache5(test_data_loader, baseline_model, fingerprint=fingerprint)

baseline_output, y_train_noise = get_all_pred_and_labels_cache5(baseline_model, train_data_loader, fingerprint=fingerprint)

baseline_confusion = sklearn.metrics.confusion_matrix(y_true=y_train_noise, y_pred=baseline_output)
channel_weights = baseline_confusion.T.copy().astype(float)
channel_weights /= channel_weights.sum(axis=1, keepdims=True)
channel_weights = np.log(channel_weights + 1e-8)
channel_weights = torch.from_numpy(channel_weights)
channel_weights = channel_weights.float()

noisemodel = Channel2D(input_dim=2, output_dim=2, theta=channel_weights.to(device))
noise_optimizer = torch.optim.Adam(noisemodel.parameters(), lr=1e-3)

print("noisy channel finished.")
early_stopping = EarlyStopping(patience=patience, verbose=True)

# noisy model train and test
for epoch in tqdm(range(epochs)):
    hybrid_train_mnist(train_data_loader, baseline_model, noisemodel, optim, noise_optimizer, criterion, BETA=BETA, fingerprint=fingerprint)
    hybrid_model = HybridModel(baseline_model, noisemodel)
    av_val_loss_baseline = validation_loss_cache5(baseline_model, val_data_loader, criterion, device, fingerprint=fingerprint)
    av_val_loss_noise_model = validation_loss_cache5(hybrid_model, val_data_loader, criterion, device, fingerprint=fingerprint)
    validation_loss = BETA * av_val_loss_baseline + (1 - BETA) * av_val_loss_noise_model
    early_stopping(validation_loss, baseline_model)
    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        early_stopping.load_best_model(baseline_model)
        break

print("Stats for Hybrid Noise Adaptive model")
accuracy, precision, recall, f1, auc_roc = test_cache5(test_data_loader, baseline_model, fingerprint=fingerprint)
print("Finished hybrid training.")


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
plot_name = "results_pickle"
plot_name = f"/nethome/pjajoria/Github/Tox21Noisy/outputs/result_pickles/{plot_name}_{current_time}.pkl"
