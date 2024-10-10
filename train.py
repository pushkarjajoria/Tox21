import random
import sklearn
from torch.utils.data import DataLoader

# from data_prep import FingerprintDataset
from data_prep_transformer import SmilesDataset
from eval import get_all_pred_and_labels
from tqdm import tqdm
import torch
import numpy as np
from model import MolPropPredictor, NoiseLayer, MolPropPredictorMolFormer, Channel, HybridModel
from plots import plot_comparison_figure
from utils import hybrid_train, test, NoisedDataset, calculate_positive_percentage, \
    split_smile_dataset_train_validation, validation_loss, EarlyStopping, validate_model, validation_loss_tox21

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

patience = 5
BETA = 0.0

# Dataset preparation
train_dataset = SmilesDataset('/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/tox21/train.smi')
test_dataset = SmilesDataset('/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/tox21/test.smi')
fingerprint = False if type(train_dataset) == SmilesDataset else True

# train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi')
# test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi')
# Metrics lists
multiple_seed_baseline_accuracy = []
multiple_seed_noise_layer_accuracy = []
multiple_seed_baseline_precision = []
multiple_seed_noise_layer_precision = []
multiple_seed_baseline_recall = []
multiple_seed_noise_layer_recall = []
multiple_seed_baseline_f1 = []
multiple_seed_noise_layer_f1 = []

num_of_seeds = 3
seeds = [42 + i for i in range(num_of_seeds)]


# Using the noised dataset for training
NOISE_LEVELS = np.linspace(0.25, 0.5, 6)
for i, seed in enumerate(seeds):
    print(f"Running for seed {i+1}/{len(seeds)}")
    # Metrics lists
    baseline_accuracy = []
    noise_layer_accuracy = []
    baseline_precision = []
    noise_layer_precision = []
    baseline_recall = []
    noise_layer_recall = []
    baseline_f1 = []
    noise_layer_f1 = []

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    for NOISE_LEVEL in NOISE_LEVELS:
        print(f"Training for Noise level {NOISE_LEVEL*100}%")
        train_dataset = NoisedDataset(original_dataset=train_dataset, noise_level=NOISE_LEVEL)
        train_data_loader, valid_data_loader = split_smile_dataset_train_validation(train_dataset)
        # train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        # TODO: Validation Set
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
        # baseline_model = MolPropPredictorMolFormer().to(device)
        baseline_model = MolPropPredictorMolFormer().to(device)
        optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)  # Use CrossEntropyLoss

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
                labels = batch['label'].long().to(device)  # Labels should be of type long for CrossEntropyLoss
                optim.zero_grad()
                output = baseline_model(x)
                loss = criterion(output, labels)
                loss.backward()
                optim.step()
                running_loss += loss.item()

            av_val_loss = validation_loss_tox21(baseline_model, valid_data_loader, criterion, device, fingerprint=fingerprint)
            print(f'Epoch: {epoch + 1}/{epochs}, '
                  f'Training Loss: {running_loss/len(train_data_loader):.4f}, '
                  f'Validation Loss: {av_val_loss:.4f}, ')

            # Check early stopping
            early_stopping(av_val_loss, None)

            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        print(f"Baseline {NOISE_LEVEL*100}% Noise")
        bl_accuracy, bl_precision, bl_recall, bl_f1 = test(test_data_loader, baseline_model, fingerprint=fingerprint)

        baseline_output, y_train_noise = get_all_pred_and_labels(baseline_model, train_data_loader, fingerprint=fingerprint)
        baseline_confusion = sklearn.metrics.confusion_matrix(y_true=y_train_noise, y_pred=baseline_output)

        channel_weights = baseline_confusion.T.copy().astype(float)
        channel_weights /= channel_weights.sum(axis=1, keepdims=True)
        channel_weights = np.log(channel_weights + 1e-8)
        channel_weights = torch.from_numpy(channel_weights)
        channel_weights = channel_weights.float()
        noisemodel = Channel(input_dim=2, output_dim=2, theta=channel_weights.to(device))
        noise_optimizer = torch.optim.Adam(noisemodel.parameters(), lr=1e-3)

        print("noisy channel finished.")
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # noisy model train and test
        for epoch in tqdm(range(epochs)):
            hybrid_train(train_data_loader, baseline_model, noisemodel, optim, noise_optimizer, criterion, BETA=BETA, fingerprint=fingerprint)
            hybrid_model = HybridModel(baseline_model, noisemodel)
            av_val_loss_baseline = validation_loss_tox21(baseline_model, valid_data_loader, criterion, device, fingerprint=fingerprint)
            av_val_loss_noise_model = validation_loss_tox21(hybrid_model, valid_data_loader, criterion, device, fingerprint=fingerprint)
            validation_loss = BETA * av_val_loss_baseline + (1 - BETA) * av_val_loss_noise_model
            early_stopping(validation_loss, None)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        print(f"After hybrid, test acc {NOISE_LEVEL*100}% Noise: ")
        accuracy, precision, recall, f1 = test(test_data_loader, baseline_model, fingerprint=fingerprint)
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

    multiple_seed_baseline_accuracy.append(baseline_accuracy)
    multiple_seed_noise_layer_accuracy.append(noise_layer_accuracy)
    multiple_seed_baseline_precision.append(baseline_precision)
    multiple_seed_noise_layer_precision.append(noise_layer_precision)
    multiple_seed_baseline_recall.append(baseline_recall)
    multiple_seed_noise_layer_recall.append(noise_layer_recall)
    multiple_seed_baseline_f1.append(baseline_f1)
    multiple_seed_noise_layer_f1.append(noise_layer_f1)

# Hyperparameter dictionary
model_info = {
    "Patience": patience,
    "Learning Rate": 1e-3,
    "Epochs": epochs,
    "Beta": BETA,
    "Comments": "TOX21\n Fingerprint \n 3Layers"
}

# Plot results
plot_comparison_figure(
    noise_levels=NOISE_LEVELS,
    plot_name="HPC-Tox21-Molformer-Finetuning-with-NAL",
    baseline_accuracy=multiple_seed_baseline_accuracy,
    noise_layer_accuracy=multiple_seed_noise_layer_accuracy,
    baseline_precision=multiple_seed_baseline_precision,
    noise_layer_precision=multiple_seed_noise_layer_precision,
    baseline_recall=multiple_seed_baseline_recall,
    noise_layer_recall=multiple_seed_noise_layer_recall,
    baseline_f1=multiple_seed_baseline_f1,
    noise_layer_f1=multiple_seed_noise_layer_f1,
    model_info=model_info
)
