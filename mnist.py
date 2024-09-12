import torchvision
from torch.utils.data import random_split

from eval import get_all_pred_and_labels_mnist
from tqdm import tqdm
import torch
import numpy as np

from logger import custom_print
from model import NoiseLayer, MNISTClassifier
from plots import plot_comparison_figure
from utils import test, get_class_distribution_and_weights, test_mnist, hybrid_train_mnist, \
    NoisedMNISTDataset, EarlyStopping, train_model_with_early_stopping, validate_model

# Backup the original print function to use later
built_in_print = print

# Override the default print with the custom print
print = custom_print

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

# Dataset preparation

batch_size_train = 64
batch_size_test = 1000

train_dataset = torchvision.datasets.MNIST('./benchmark_datasets/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_dataset = torchvision.datasets.MNIST('./benchmark_datasets/mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_data_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size_test, shuffle=True)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
val_data_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size_test, shuffle=False)  # Validation data loader
baseline_accuracy = []
noise_layer_accuracy = []

baseline_precision = []
noise_layer_precision = []

baseline_recall = []
noise_layer_recall = []

baseline_f1 = []
noise_layer_f1 = []

# Using the noised dataset for training
NOISE_LEVELS = np.linspace(0.2, 0.5, 7)
for NOISE_LEVEL in NOISE_LEVELS:
    print(f"Training for Noise level {NOISE_LEVEL*100}%")

    noised_dataset = NoisedMNISTDataset(original_dataset=train_subset, noise_level=NOISE_LEVEL)

    train_data_loader = torch.utils.data.DataLoader(noised_dataset, batch_size=batch_size_train, shuffle=True)
    print(f"Training(NOISED {NOISE_LEVEL}%) Dataset Distribution Table")
    class_weights, total_samples = get_class_distribution_and_weights(noised_dataset, device)
    print("-"*40)
    print("Testing Dataset Distribution Table")
    _, _ = get_class_distribution_and_weights(test_dataset, device)

    # Model setup
    baseline_model = MNISTClassifier().to(device)
    optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    epochs = 50

    train_model_with_early_stopping(
        model=baseline_model,
        train_data_loader=train_data_loader,
        valid_data_loader=val_data_loader,  # Pass your validation data loader here
        criterion=criterion,
        optimizer=optim,
        device=device,
        epochs=epochs,  # Number of epochs
        early_stopping_patience=3  # Patience for early stopping
    )

    print(f"Baseline {NOISE_LEVEL*100}% Noise")
    bl_accuracy, bl_precision, bl_recall, bl_f1 = test_mnist(test_data_loader, baseline_model)

    baseline_output, y_train_noise = get_all_pred_and_labels_mnist(baseline_model, train_data_loader)
    baseline_confusion = np.zeros((10, 10))
    for n, p in zip(y_train_noise, baseline_output):
        baseline_confusion[p, n] += 1.

    channel_weights = baseline_confusion.copy()
    channel_weights /= channel_weights.sum(axis=1, keepdims=True)
    channel_weights = np.log(channel_weights + 1e-8)
    channel_weights = torch.from_numpy(channel_weights)
    channel_weights = channel_weights.float()
    noisemodel = NoiseLayer(theta=channel_weights.to(device), k=10)
    noise_optimizer = torch.optim.Adam(noisemodel.parameters(),
                                 lr=1e-3)

    print("noisy channel finished.")
    # noisy model train and test

    for epoch in tqdm(range(epochs)):
        hybrid_train_mnist(train_data_loader, baseline_model, noisemodel, optim, noise_optimizer, criterion)

    print(f"After hybrid, test acc {NOISE_LEVEL*100}% Noise: ")
    accuracy, precision, recall, f1 = test_mnist(test_data_loader, baseline_model)
    print("Finished hybrid training.")

    baseline_accuracy.append(bl_accuracy)
    noise_layer_accuracy.append(accuracy)
    baseline_precision.append(bl_precision)
    noise_layer_precision.append(precision)
    baseline_recall.append(bl_recall)
    noise_layer_recall.append(recall)
    baseline_f1.append(bl_f1)
    noise_layer_f1.append(f1)

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