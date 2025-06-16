import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import random


def smiles_to_morgan_fingerprint(smile, radius=2, n_bits=2048):
    """Convert SMILES string to Morgan fingerprint."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros(n_bits)  # Return a zero vector for invalid SMILES
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)


def get_cv_splits(cv_df, num_folds, fold_col):
    train_indices = []
    val_indices = []
    for fold_idx in range(num_folds):
        test_fold = fold_idx

        test_idx = cv_df[cv_df[fold_col] == f'Fold_{test_fold}'].index.to_list()
        train_idx = cv_df[cv_df[fold_col] != f'Fold_{test_fold}'].index.to_list()

        train_indices.append(train_idx)
        val_indices.append(test_idx)
    return train_indices, val_indices


def create_train_test_val_splits(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, index_col=0)

    # Convert SMILES to Morgan fingerprints
    data["morgan_fp"] = list(map(smiles_to_morgan_fingerprint, data['smiles'].values))

    # Define the fold splits
    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]
    val_fold = "Fold_4"
    test_fold = [f"Fold_{i}" for i in [8, 9]]

    # Create train, validation, and test sets based on the 'DataSAIL_10f' column
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]
    val_data = data[data["DataSAIL_10f"] == val_fold]
    test_data = data[data["DataSAIL_10f"].isin(test_fold)]

    # Extract X (Morgan fingerprints) and y (class labels)
    X_train = np.array(train_data["morgan_fp"].to_list())
    y_train = train_data["class"].values

    X_val = np.array(val_data["morgan_fp"].to_list())
    y_val = val_data["class"].values

    X_test = np.array(test_data["morgan_fp"].to_list())
    y_test = test_data["class"].values

    return CacheDataset(X_train, y_train), CacheDataset(X_val, y_val), CacheDataset(X_test, y_test)


class FingerprintDataset(Dataset):
    def __init__(self, file_path=None, n_bits=2048, fingerprints=None, labels=None, masks=None):
        self.n_bits = n_bits
        if fingerprints is not None and labels is not None and masks is not None:
            self.x = fingerprints
            self.labels = labels
            self.masks = masks
        elif file_path is not None:
            self.masks = []
            self.smiles = []
            self.labels = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        "Headers"
                        continue
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        smile = parts[1]
                        label = [int(float(x)) if x in ["0.0", "1.0"] else float('nan') for x in parts[2:]]
                        mask = self._compute_label_mask(label)
                        self.smiles.append(smile)
                        self.labels.append(label)
                        self.masks.append(mask)
            self.x = np.array([smiles_to_morgan_fingerprint(s, n_bits=n_bits) for s in self.smiles])
        else:
            raise ValueError('file_path must be specified')

        self.labels = np.array(self.labels)

    @staticmethod
    def _compute_label_mask(label):
        # Creates a mask for labels where it is true if the label is 0 or 1 and False if it is Float('nan')
        return [x == x for x in label]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': torch.tensor(self.x[idx], dtype=torch.float),
                'label': torch.tensor(self.labels[idx], dtype=torch.float),
                'mask': torch.tensor(self.masks[idx], dtype=torch.float)}

    def fit_transform_pca(self, n_components=100, pca=None):
        if pca is None:
            pca = PCA(n_components=n_components)
            pca.fit(self.x)
        self.x = pca.transform(self.x)
        return pca


class CacheDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.labels = y

    def __getitem__(self, idx):
        return {'x': torch.tensor(self.x[idx], dtype=torch.float),
                'label': torch.tensor(self.labels[idx], dtype=torch.float)}

    def __len__(self):
        return len(self.x)

    def fit_transform_pca(self, n_components=100, pca=None):
        if pca is None:
            pca = PCA(n_components=n_components)
            pca.fit(self.x)
        self.x = pca.transform(self.x)
        return pca


if __name__ == "__main__":
    data_path = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
    train, val, test = create_train_test_val_splits(data_path)
    print(len("Kamehameha"))

