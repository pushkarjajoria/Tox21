import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader


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
        self.x = self.dataset.x

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
        return {'x': self.dataset[idx]['x'], 'label': torch.tensor(self.noised_labels[idx], dtype=torch.float)}


def test(test_loader, model):
    all_preds = []
    all_labels = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].float().to(device)
            labels = batch['label'].long().to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1).cpu().numpy()  # Apply threshold for binary classification
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


def accuracy(output, target):
    correct = 0
    _, pred = output.max(1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / target.size(0)
    return acc


def hybrid_train(train_loader, model, noisemodel, optimizer, noise_optimizer, criterion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BETA = 0.8
    model.train()
    noisemodel.train()
    for batch in train_loader:
        x = batch['x'].float().to(device)
        labels = batch['label'].long().to(device)
        # set all gradient to zero
        optimizer.zero_grad()
        noise_optimizer.zero_grad()
        # forward propagation
        out = model(x)
        predictions = noisemodel(out)
        # calculate loss and acc
        baseline_loss = criterion(out, labels)
        noise_model_loss = criterion(predictions, labels)
        loss = (1 - BETA) * baseline_loss + BETA * noise_model_loss

        # back propagation
        loss.backward()
        # update the parameters (weights and biases)
        optimizer.step()
        noise_optimizer.step()


# Calculate and print percentage of positive test results
def calculate_positive_percentage(dataset):
    all_labels = []
    for batch in DataLoader(dataset, batch_size=64, shuffle=False):
        labels = batch['label'].numpy()
        all_labels.extend(labels)
    all_labels = np.array(all_labels)
    positive_percentage = np.mean(all_labels == 1) * 100
    return positive_percentage
