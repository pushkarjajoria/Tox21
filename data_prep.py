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


class NoisedDataset(Dataset):
    def __init__(self, original_dataset, noise_level=0.1):
        """
        Args:
            original_dataset (Dataset): The original dataset.
            noise_level (float): The proportion of labels to be noised. Default is 0.1 (10%).
        """
        self.dataset = original_dataset
        self.noise_level = noise_level
        self.noised_labels = self._create_noised_labels()
        self.fingerprints = self.dataset.fingerprints

    def _create_noised_labels(self):
        """
        Applies noise to the labels while preserving class distribution.
        """
        original_labels = [self.dataset[i]['label'] for i in range(len(self.dataset))]

        class_0_indices = [i for i, label in enumerate(original_labels) if label == 0.]
        class_1_indices = [i for i, label in enumerate(original_labels) if label == 1.]

        noised_labels = original_labels.copy()

        num_of_samples_to_noise = int(len(class_1_indices) * self.noise_level)

        # Apply noise to a percentage of the 0s
        noise_class_0_indices = random.sample(class_0_indices, num_of_samples_to_noise)
        for i in noise_class_0_indices:
            noised_labels[i] = 1  # Flip 0 to 1

        # Apply noise to a percentage of the 1s
        noise_class_1_indices = random.sample(class_1_indices, num_of_samples_to_noise)
        for i in noise_class_1_indices:
            noised_labels[i] = 0  # Flip 1 to 0

        return noised_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {'fingerprint': self.dataset[idx]['fingerprint'],
                    'label': torch.tensor(self.noised_labels[idx], dtype=torch.float)}


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
        self.fingerprints = np.array([smiles_to_morgan_fingerprint(s, n_bits=n_bits) for s in self.smiles])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {'fingerprint': torch.tensor(self.fingerprints[idx], dtype=torch.float),
                'label': torch.tensor(self.labels[idx], dtype=torch.float)}


# Calculate and print percentage of positive test results
def calculate_positive_percentage(dataset):
    all_labels = []
    for batch in DataLoader(dataset, batch_size=64, shuffle=False):
        labels = batch['label'].numpy()
        all_labels.extend(labels)
    all_labels = np.array(all_labels)
    positive_percentage = np.mean(all_labels == 1) * 100
    return positive_percentage


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
