import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from model import Molformer


class SmilesDataset(Dataset):
    def __init__(self, file_path, batch_size=32):
        self.batch_size = batch_size
        self.smiles = []
        self.labels = []
        self.x = []  # Reverting the name from "embeddings" to "x"

        molformer = Molformer()

        # Load SMILES and labels
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    smile = parts[0]
                    label = int(parts[1])
                    self.smiles.append(smile)
                    self.labels.append(label)

        # Process SMILES in batches
        for i in range(0, len(self.smiles), self.batch_size):
            batch_smiles = self.smiles[i:i + self.batch_size]
            with torch.no_grad():
                batch_x = molformer(batch_smiles)  # Use "batch_x" instead of "batch_embeddings"
            self.x.append(batch_x.cpu())

        # Flatten the list of tensors into a single tensor
        self.x = torch.cat(self.x, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {'x': self.x[idx],  # Return "x" instead of "embedding"
                'label': self.labels[idx]}
def read_csv_property_file(filepath, cols_to_read):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # Extract the specified columns
    data = df.iloc[:, cols_to_read].values.T

    return data


def save_transformer_data(filepath, smiles, labels):
    # Combine smiles and labels into a DataFrame
    data = pd.DataFrame({'smiles': smiles})
    label_cols = {f'label_{i}': labels[:, i] for i in range(labels.shape[1])}
    label_df = pd.DataFrame(label_cols)

    final_df = pd.concat([data, label_df], axis=1)

    # Save the DataFrame to a CSV file
    final_df.to_csv(filepath, index=False)


if __name__ == "__main__":
    # Use the new function to read data from CSV
    data = read_csv_property_file('benchmark_datasets/tox21/tox21.csv',
                                  cols_to_read=[13] + list(range(0, 12)))

    smiles = data[0]
    labels = np.array(data[1:])

    # Replace empty strings with NaN
    labels[labels == ''] = np.nan

    # Convert to float to handle NaN, then replace NaN with 9 and convert to int
    # Replace empty strings with NaN
    labels[labels == ''] = np.nan

    # Convert to float to handle NaN, then replace NaN with 9 and convert to int
    labels = labels.astype(float)
    labels = np.nan_to_num(labels, nan=9).astype(int)

    # Filter out datapoints where label[0] is 9
    valid_indices = labels[0] != 9
    smiles = np.array(smiles)[valid_indices]
    labels = labels[:, valid_indices]

    labels = labels.T
    X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                        random_state=42)

    # Use the new function to save the data with "transformer" in the filename
    save_transformer_data('./benchmark_datasets/tox21/train_transformer.csv', X_train, y_train)
    save_transformer_data('./benchmark_datasets/tox21/test_transformer.csv', X_test, y_test)
