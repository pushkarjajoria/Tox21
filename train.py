import copy
import pickle
import random
from datetime import datetime

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_prep import FingerprintDataset
from data_prep_transformer import SmilesDataset
from eval import get_all_pred_and_labels
from model import Channel, HybridModel, Tox21Predictor
from plots import plot_results
from utils import hybrid_train, test, NoisedDataset, calculate_positive_percentage, \
    EarlyStopping, validation_loss_tox21, \
    iterative_validation_split

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)

patience = 5
BETA = 0.0

# Dataset preparation
train_file = '/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/tox21/tox21_10k_data_all_duplicates_merged.csv'
train_dataset = FingerprintDataset(train_file)
test_file = '/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/tox21/tox21_10k_challenge_test_duplicates_merged.csv'
test_dataset_original = FingerprintDataset(test_file)
fingerprint = False if type(train_dataset) == SmilesDataset else True

# Input about PCA Transformation
pca_transform_data = True
n_pca_components = 1024

# Metrics lists
multiple_seed_baseline_accuracy = []
multiple_seed_noise_layer_accuracy = []
multiple_seed_baseline_precision = []
multiple_seed_noise_layer_precision = []
multiple_seed_baseline_recall = []
multiple_seed_noise_layer_recall = []
multiple_seed_baseline_f1 = []
multiple_seed_noise_layer_f1 = []
multiple_seed_baseline_auc = []
multiple_seed_noise_layer_auc = []

num_of_seeds = 3
seeds = [42 + i for i in range(num_of_seeds)]
do_once = True

# Using the noised dataset for training
NOISE_LEVELS = np.linspace(0.05, 0.5, 10)
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
    baseline_roc_auc = []
    noise_layer_roc_auc = []

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    for NOISE_LEVEL in NOISE_LEVELS:
        print(f"Training for Noise level {NOISE_LEVEL*100}%")
        train_dataset = NoisedDataset(original_dataset=train_dataset, noise_level=NOISE_LEVEL)
        train_data_loader, valid_data_loader, pca = iterative_validation_split(train_dataset,
                                                                               pca_transformation=pca_transform_data,
                                                                               n_pca_components=n_pca_components)
        # train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_dataset = copy.copy(test_dataset_original)
        if pca_transform_data:
            test_dataset.fit_transform_pca(pca=pca)

        test_data_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

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
        baseline_model = Tox21Predictor(input_size=input_size, output_size=12 * 2, seed=seed).to(device)
        optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-5)
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

        print(f"Baseline {NOISE_LEVEL*100}% Noise")
        accuracy_task_map, precision_task_map, recall_task_map, f1_task_map, roc_auc_task_map \
            = test(test_data_loader, baseline_model, fingerprint=fingerprint, verbose=True)

        baseline_output, y_train_noise = get_all_pred_and_labels(baseline_model, train_data_loader, fingerprint=fingerprint)
        channel_weights_per_task = []
        for i in range(num_tasks):
            baseline_confusion = sklearn.metrics.confusion_matrix(y_true=y_train_noise[:, i], y_pred=baseline_output[:, i])
            channel_weights = baseline_confusion.T.copy().astype(float)
            channel_weights /= channel_weights.sum(axis=1, keepdims=True)
            channel_weights = np.log(channel_weights + 1e-8)
            channel_weights = torch.from_numpy(channel_weights)
            channel_weights = channel_weights.float()
            channel_weights_per_task.append(channel_weights)
        channel_weights_per_task = torch.stack(channel_weights_per_task, dim=0)
        # Start Here. The Channel layer now takes 12 channel weights, 1 for each task and then uses them accordingly in
        #   the forward pass
        noisemodel = Channel(input_dim=2, output_dim=2, theta=channel_weights_per_task.to(device))
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
            early_stopping(validation_loss, baseline_model)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                early_stopping.load_best_model(baseline_model)
                break

        print(f"After hybrid, test acc {NOISE_LEVEL*100}% Noise: ")
        accuracy, precision, recall, f1, auc_roc = test(test_data_loader, baseline_model, fingerprint=fingerprint, verbose=True)
        print("Finished hybrid training.")

        # Collect metrics
        # baseline_accuracy.append(bl_accuracy)
        # noise_layer_accuracy.append(accuracy)
        # baseline_precision.append(bl_precision)
        # noise_layer_precision.append(precision)
        # baseline_recall.append(bl_recall)
        # noise_layer_recall.append(recall)
        baseline_f1.append(f1_task_map)
        noise_layer_f1.append(f1)
        baseline_roc_auc.append(roc_auc_task_map)
        noise_layer_roc_auc.append(auc_roc)

    # multiple_seed_baseline_accuracy.append(baseline_accuracy)
    # multiple_seed_noise_layer_accuracy.append(noise_layer_accuracy)
    # multiple_seed_baseline_precision.append(baseline_precision)
    # multiple_seed_noise_layer_precision.append(noise_layer_precision)
    # multiple_seed_baseline_recall.append(baseline_recall)
    # multiple_seed_noise_layer_recall.append(noise_layer_recall)
    multiple_seed_baseline_f1.append(baseline_f1)
    multiple_seed_noise_layer_f1.append(noise_layer_f1)
    multiple_seed_baseline_auc.append(baseline_roc_auc)
    multiple_seed_noise_layer_auc.append(noise_layer_roc_auc)

# Hyperparameter dictionary
model_info = {
    "Patience": patience,
    "Learning Rate": 1e-3,
    "Epochs": epochs,
    "Beta": BETA,
    "Comments": "TOX21\n Fingerprint \n 3Layers"
}

# Plot results
# plot_comparison_figure(
#     noise_levels=NOISE_LEVELS,
#     plot_name="HPC-Tox21-Molformer-Finetuning-with-NAL",
#     baseline_accuracy=multiple_seed_baseline_accuracy,
#     noise_layer_accuracy=multiple_seed_noise_layer_accuracy,
#     baseline_precision=multiple_seed_baseline_precision,
#     noise_layer_precision=multiple_seed_noise_layer_precision,
#     baseline_recall=multiple_seed_baseline_recall,
#     noise_layer_recall=multiple_seed_noise_layer_recall,
#     baseline_f1=multiple_seed_baseline_f1,
#     noise_layer_f1=multiple_seed_noise_layer_f1,
#     model_info=model_info
# )

results_dict = {
    "baseline_f1": multiple_seed_baseline_f1,
    "noise_layer_f1": multiple_seed_noise_layer_f1,
    "baseline_roc_auc": multiple_seed_baseline_auc,
    "noise_layer_roc_auc": multiple_seed_noise_layer_auc,
    "num_seed": num_of_seeds,
    "noise_levels": NOISE_LEVELS
}

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
plot_name = "results_pickle"
plot_name = f"/nethome/pjajoria/Github/Tox21Noisy/outputs/result_pickles/{plot_name}_{current_time}.pkl"
with open(plot_name, "wb") as f:
    pickle.dump(results_dict, f)

plot_results(results_dict)
