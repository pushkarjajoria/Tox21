import pickle
import numpy as np
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import get_tokens
from sklearn.model_selection import train_test_split
from openchem.data.utils import save_smiles_property_file
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


class FingerprintDataset(Dataset):
    def __init__(self, file_path, n_bits=2048):
        self.n_bits = n_bits
        self.smiles = []
        self.labels = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    smile = parts[0]
                    label = int(parts[1])
                    self.smiles.append(smile)
                    self.labels.append(label)
        self.x = np.array([smiles_to_morgan_fingerprint(s, n_bits=n_bits) for s in self.smiles])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {'x': torch.tensor(self.fingerprints[idx], dtype=torch.float),
                'label': torch.tensor(self.labels[idx], dtype=torch.float)}


if __name__ == "__main__":

    data = read_smiles_property_file('../OpenChem/benchmark_datasets/tox21/tox21.csv',
                                     cols_to_read=[13] + list(range(0,12)))
    smiles = data[0]
    labels = np.array(data[1:])

    labels[np.where(labels=='')] = '9'
    labels = labels.astype(int)

    # Filter out datapoints where label[0] is 9
    valid_indices = labels[0] != 9
    smiles = np.array(smiles)[valid_indices]
    labels = labels[:, valid_indices]

    labels = labels.T

    tokens, _, _ = get_tokens(smiles)
    tokens = tokens + ' '
    with open("benchmark_datasets/tox21/tokens.txt", 'wb') as f:
        pickle.dump(tokens, f)
    X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                        random_state=42)

    save_smiles_property_file('./benchmark_datasets/tox21/train.smi', X_train, y_train)
    save_smiles_property_file('./benchmark_datasets/tox21/test.smi', X_test, y_test)
