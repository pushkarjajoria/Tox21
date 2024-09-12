import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from eval import get_all_pred_and_labels_mnist
from logger import custom_print
from model import NoiseLayer, MNISTClassifier, HybridModel
from plots import plot_comparison_figure
from utils import (test, get_class_distribution_and_weights, test_mnist, hybrid_train_mnist,
                   NoisedMNISTDataset, EarlyStopping, train_model_with_early_stopping, validate_model)


# Load MNIST data
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# Parameters
batch_size_train = 64
batch_size_test = 1000
patience = 4
img_color, img_rows, img_cols = 1, 28, 28
img_size = img_color * img_rows * img_cols
seed = 42

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = load_data(path="benchmark_datasets/mnist.npz")

print('MNIST training data set label distribution', np.bincount(y_train))
print('Test distribution', np.bincount(y_test))

X_train = X_train.reshape(X_train.shape[0], img_size).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], img_size).astype('float32') / 255.

tensor_x_test = torch.Tensor(X_test)
tensor_y_test = torch.Tensor(y_test)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size_test)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Noise setup
perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])
noise = perm[y_train]

np.random.seed(seed)
random.seed(seed)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

# Metrics lists
baseline_accuracy = []
noise_layer_accuracy = []
baseline_precision = []
noise_layer_precision = []
baseline_recall = []
noise_layer_recall = []
baseline_f1 = []
noise_layer_f1 = []

# Training with different noise levels
NOISE_LEVELS = np.linspace(0.2, 0.5, 7)
for NOISE_LEVEL in NOISE_LEVELS:
    print(f"Training for Noise level {NOISE_LEVEL * 100}%")

    # Apply noise to the dataset
    _, noise_idx = next(
        iter(StratifiedShuffleSplit(n_splits=1, test_size=NOISE_LEVEL, random_state=seed).split(X_train, y_train)))
    y_train_noise = y_train.copy()
    y_train_noise[noise_idx] = noise[noise_idx]
    print(f"Actual Noise {(1. - np.mean(y_train_noise == y_train)) * 100}%")

    # Split data into training and validation sets
    train_idx, val_idx = next(
        iter(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(X_train, y_train_noise)))
    X_train_train, y_train_train = X_train[train_idx], y_train_noise[train_idx]
    X_train_val, y_train_val = X_train[val_idx], y_train_noise[val_idx]

    # Model setup
    baseline_model = MNISTClassifier().to(device)
    optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epochs = 50

    # Prepare data loaders
    tensor_x_train = torch.Tensor(X_train_train)
    tensor_y_train = torch.Tensor(y_train_train)
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size_train)

    tensor_x_val = torch.Tensor(X_train_val)
    tensor_y_val = torch.Tensor(y_train_val)
    val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size_test)

    # Train baseline model
    train_model_with_early_stopping(
        model=baseline_model,
        train_data_loader=train_data_loader,
        valid_data_loader=val_data_loader,
        criterion=criterion,
        optimizer=optim,
        device=device,
        epochs=epochs,
        early_stopping_patience=patience
    )

    print(f"Baseline {NOISE_LEVEL * 100}% Noise")
    bl_accuracy, bl_precision, bl_recall, bl_f1 = test_mnist(test_data_loader, baseline_model)

    # Compute confusion matrix for channel weights
    baseline_output, y_train_noise = get_all_pred_and_labels_mnist(baseline_model, train_data_loader)
    baseline_confusion = np.zeros((10, 10))
    for n, p in zip(y_train_noise, baseline_output):
        baseline_confusion[p, n] += 1.

    # Compute channel weights TODO: Check this part of the code!
    channel_weights = baseline_confusion.copy()
    channel_weights /= channel_weights.sum(axis=1, keepdims=True)
    channel_weights = np.log(channel_weights + 1e-8)
    channel_weights = torch.from_numpy(channel_weights).float()

    # Setup noisy model
    noisemodel = NoiseLayer(theta=channel_weights.to(device), k=10)
    noise_optimizer = torch.optim.Adam(noisemodel.parameters(), lr=1e-3)

    print("Noisy channel finished.")

    # Hybrid model training
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in tqdm(range(epochs)):
        hybrid_train_mnist(train_data_loader, baseline_model, noisemodel, optim, noise_optimizer, criterion)
        hybrid_model = HybridModel(baseline_model, noisemodel)
        val_accuracy, val_precision, val_recall, val_f1, av_val_loss = validate_model(baseline_model, val_data_loader,
                                                                                      criterion, device)
        early_stopping(av_val_loss, baseline_model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    print(f"After hybrid, test acc {NOISE_LEVEL * 100}% Noise: ")
    accuracy, precision, recall, f1 = test_mnist(test_data_loader, baseline_model)
    print("Finished hybrid training.")

    # Collect metrics
    baseline_accuracy.append(bl_accuracy)
    noise_layer_accuracy.append(accuracy)
    baseline_precision.append(bl_precision)
    noise_layer_precision.append(precision)
    baseline_recall.append(bl_recall)
    noise_layer_recall.append(recall)
    baseline_f1.append(bl_f1)
    noise_layer_f1.append(f1)

# Plot results
plot_comparison_figure(
    noise_levels=NOISE_LEVELS,
    baseline_accuracy=baseline_accuracy,
    noise_layer_accuracy=noise_layer_accuracy,
    baseline_precision=baseline_precision,
    noise_layer_precision=noise_layer_precision,
    baseline_recall=baseline_recall,
    noise_layer_recall=noise_layer_recall,
    baseline_f1=baseline_f1,
    noise_layer_f1=noise_layer_f1
)
