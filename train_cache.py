import random
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_prep import create_train_test_val_splits
from data_prep_transformer import SmilesDataset
from utils import test_cache5, validation_loss_cache5
from model import Cache5AntagonistPredictor, IFMEncoder, SpikingCache5AntagonistPredictor
from utils import calculate_positive_percentage, EarlyStopping

# ---------------- Helper Functions ---------------- #


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_datasets(data_path, batch_size, val_batch_size, pca_transform_data=False, n_pca_components=1024):
    train_dataset, val_dataset, test_dataset = create_train_test_val_splits(data_path)
    fingerprint = not isinstance(train_dataset, SmilesDataset)

    if pca_transform_data:
        pca = train_dataset.fit_transform_pca(n_components=n_pca_components)
        val_dataset.fit_transform_pca(pca=pca)
        test_dataset.fit_transform_pca(pca=pca)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, fingerprint


def build_model(input_size, seed, device, class_weights, with_embedder=False, embedder=None, activation=None, model=None):
    if model is None:
        model = Cache5AntagonistPredictor(
            input_size=input_size,
            output_size=2,
            seed=seed,
            embedding=with_embedder,
            embedder=embedder,
            activation=activation
        ).to(device)
    else:
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=class_weights).to(device)
    return model, optimizer, criterion


def train_model(model, optimizer, criterion, train_loader, val_loader, device, fingerprint, epochs=50, patience=5, verbose=False):
    early_stopping = EarlyStopping(patience=patience)
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0
        for batch in train_loader:
            x = batch['x'].float().to(device) if fingerprint else batch['x']
            labels = batch['label'].long().to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validation_loss_cache5(model, val_loader, criterion, device, fingerprint=fingerprint)
        if verbose:
            print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model=model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Restoring best model.")
            early_stopping.load_best_model(model)
            break
    return model


def evaluate_model(model, test_loader, fingerprint):
    return test_cache5(test_loader, model, fingerprint=fingerprint)


class MajorityClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.stack([torch.ones(x.shape[0]), torch.zeros(x.shape[0])], dim=1).to(x.device)


# ---------------- Main Pipeline ---------------- #
def main(with_embedder=False, embedder=None, patience=5, activation=torch.relu, verbose=False, model=None):
    device = get_device()
    print(f"Running the model on {device}")
    print(torch.version.cuda)

    seed = 42
    set_random_seed(seed)

    # Configurations
    patience = patience
    batch_size = 64
    val_batch_size = 1000
    pca_transform_data = False
    n_pca_components = 1024
    data_path = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'

    # Prepare datasets and loaders
    (train_dataset, val_dataset, test_dataset,
     train_loader, val_loader, test_loader, fingerprint) = prepare_datasets(
        data_path, batch_size, val_batch_size, pca_transform_data, n_pca_components
    )

    # Calculate and display class statistics
    train_positive_percentage = calculate_positive_percentage(train_dataset)
    test_positive_percentage = calculate_positive_percentage(test_dataset)
    if verbose:
        print(f"Len of Train: {len(train_dataset)}")
        print(f"Len of Test: {len(test_dataset)}")
        print(f"Percentage of positive results in training: {train_positive_percentage}%")
        print(f"Percentage of positive results in testing: {test_positive_percentage}%")

    positive_weight = (100 - train_positive_percentage) / 100.0
    negative_weight = train_positive_percentage / 100.0
    class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32).to(device)

    # Determine input size based on PCA setting
    input_size = n_pca_components if pca_transform_data else 2048

    # Build model
    model, optimizer, criterion = build_model(input_size, seed, device, class_weights, with_embedder, embedder, activation=activation, model=model)
    # model = MajorityClassifier()
    # Train model
    model = train_model(model, optimizer, criterion, train_loader, val_loader, device, fingerprint, epochs=50, patience=patience, verbose=verbose)

    # Evaluate model
    metrics = evaluate_model(model, test_loader, fingerprint)
    (accuracy_task_map, precision_task_map, recall_task_map,
     f1_task_map, roc_auc_task_map) = metrics

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plot_path = f"/nethome/pjajoria/Github/Tox21Noisy/outputs/result_pickles/results_pickle_{current_time}.pkl"
    return model, metrics, plot_path


if __name__ == "__main__":
    # Create an embedder instance for experiments requiring embedding
    ifm_embedder = IFMEncoder(2048, 8, 6)

    print("Experiment 1: Baseline Model (Cache5)")
    model_baseline, metrics_baseline, plot_path_baseline = main(with_embedder=False)

    print("\nExperiment 2: Model with embedder (patience=7)")
    model_with_embedder, metrics_with_embedder, plot_path_embedder = main(with_embedder=True, embedder=ifm_embedder, patience=7)

    print("\nExperiment 3: Model without embedder using tanh activation (patience=7)")
    model_no_embed_tanh, metrics_no_embed_tanh, plot_path_no_embed_tanh = main(with_embedder=False, patience=7, activation=torch.tanh)

    print("\nExperiment 4: Model without embedder using sigmoid activation (patience=7)")
    model_no_embed_sigmoid, metrics_no_embed_sigmoid, plot_path_no_embed_sigmoid = main(with_embedder=False, patience=7, activation=torch.sigmoid)

    print("\nExperiment 5: Model with embedder using tanh activation (patience=7)")
    model_embed_tanh, metrics_embed_tanh, plot_path_embed_tanh = main(with_embedder=True, embedder=ifm_embedder, patience=7, activation=torch.tanh)

    # print("\nExperiment 6: Spiking NN")
    # model = SpikingCache5AntagonistPredictor().to("cuda")
    # model_spiking, metrics_spiking, plot_path_spiking = main(patience=7, model=model)
