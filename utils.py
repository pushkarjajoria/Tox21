import random

from numpy.testing import assert_array_almost_equal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tabulate import tabulate
from torchvision.datasets import MNIST
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


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


class NoisedMNISTDataset(Dataset):
    def __init__(self, original_dataset: MNIST, noise_level=0.1):
        """
        Args:
            original_dataset (Dataset): The original dataset.
            noise_level (float): The proportion of labels to be noised. Default is 0.1 (10%).
        """
        if isinstance(original_dataset, torch.utils.data.Subset):
            self.dataset = original_dataset.dataset
        else:
            self.dataset = original_dataset
        self.noise_level = noise_level
        self.labels = self.dataset.targets
        self.noised_labels = self._create_noised_labels()

    def _create_noised_labels(self):
        """
        Applies noise to the labels based on predefined flipping rules.
        """
        original_labels = self.labels
        flip_rules = {
            1: [7],
            2: [7],
            3: [8],
            5: [6],
            7: [1],
            6: [5]
        }

        noised_labels = original_labels.detach().clone()

        # Approximate number of samples from the noisy class set to be flipped
        num_of_samples_to_noise = int((len(self.labels) * 0.6) * self.noise_level)

        # Collect all indices of the labels that match the keys of flip_rules
        indices_of_noisy_classes = [i for i, label in enumerate(original_labels) if label.detach().cpu().item() in flip_rules]

        # Randomly sample `num_of_samples_to_noise` indices to apply noise
        indices_to_noise = random.sample(indices_of_noisy_classes, num_of_samples_to_noise)

        for idx in indices_to_noise:
            current_label = original_labels[idx].item()
            # Flip the label according to the rules
            if current_label in flip_rules:
                noised_labels[idx] = random.choice(flip_rules[current_label])

        return noised_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (self.dataset[idx][0], torch.tensor(self.noised_labels[idx], dtype=torch.int))


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


def test_mnist(test_loader, model):
    all_preds = []
    all_labels = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].float().to(device)  # Use batch[0] for input images (MNIST)
            labels = batch[1].long().to(device)  # Use batch[1] for labels (MNIST)
            output = model(x)
            preds = torch.argmax(output, dim=1).cpu().numpy()  # Get predicted class
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')  # Weighted average for multi-class
    recall = recall_score(all_labels, all_preds, average='weighted')  # Weighted average
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Weighted average

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (weighted): {precision:.4f}')
    print(f'Recall (weighted): {recall:.4f}')
    print(f'F1 Score (weighted): {f1:.4f}')

    return accuracy, precision, recall, f1


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


def get_class_distribution_and_weights(dataset, device):
    all_labels = []
    for batch in DataLoader(dataset, batch_size=64, shuffle=False):
        labels = batch[1].numpy()
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels, minlength=10)  # Assuming 10 classes for MNIST
    total_samples = len(all_labels)

    # Print class distribution in a tabular format
    table = []
    for i in range(10):
        class_percentage = (class_counts[i] / total_samples) * 100
        table.append([i, class_counts[i], f"{class_percentage:.2f}%"])
    print(tabulate(table, headers=["Class", "Sample Count", "Percentage"], tablefmt="grid"))

    # Compute class weights based on distribution
    class_percentages = class_counts / total_samples
    class_weights = (1.0 / class_percentages)  # Inverse class frequency weighting
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    return class_weights, total_samples


def hybrid_train_mnist(train_loader, model, noisemodel, optimizer, noise_optimizer, criterion):
    # Include Validation dataset aswell

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BETA = 1  # Ignore baseline loss
    model.train()
    noisemodel.train()

    total_loss = 0  # Accumulator for total loss

    for batch in train_loader:
        x = batch[0].float().to(device)
        labels = batch[1].long().to(device)

        # Set all gradients to zero
        optimizer.zero_grad()
        noise_optimizer.zero_grad()

        # Forward propagation
        out = model(x)
        predictions = noisemodel(out)

        # Calculate loss
        baseline_loss = criterion(out, labels)
        noise_model_loss = criterion(predictions, labels)
        loss = (1-BETA) * baseline_loss + BETA * noise_model_loss

        # Backward propagation
        loss.backward()

        # Update the parameters (weights and biases)
        optimizer.step()
        noise_optimizer.step()

        # Accumulate loss for printing
        total_loss += loss.item()

    # Print the total loss for this epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch Loss: {avg_loss:.4f}')


def noisify_pairflip_mnist(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    print('----------Pair noise----------')
    P = np.eye(nb_classes)
    n = noise

    """mistakes:
    1 <- 7
    2 -> 7
    3 -> 8
    5 <-> 6
    """
    # 1 <- 7
    P[7, 7], P[7, 1] = 1. - n, n
    # 2 -> 7
    P[2, 2], P[2, 7] = 1. - n, n
    # 5 <-> 6
    P[5, 5], P[5, 6] = 1. - n, n
    P[6, 6], P[6, 5] = 1. - n, n
    # 3 -> 8
    P[3, 3], P[3, 8] = 1. - n, n
    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    return y_train_noisy, actual_noise, P


def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def dataset_split(train_images, train_labels, noise_rate=0.0, split_per=0.9, random_seed=42, num_classes=10):
    clean_train_labels = train_labels[:, np.newaxis]

    noisy_labels, real_noise_rate, _ = noisify_pairflip_mnist(clean_train_labels, noise=noise_rate, random_state=random_seed,
                                                            nb_classes=num_classes)
    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels


def validate_model(model, valid_data_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    # Get the metrics using the test function
    accuracy, precision, recall, f1 = test_mnist(valid_data_loader, model)

    with torch.no_grad():
        for data, target in valid_data_loader:
            data, target = data.to(device), target.long().to(device)

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_data_loader)  # Return average validation loss
    return accuracy, precision, recall, f1, avg_val_loss


def train_model_with_early_stopping(
        model, train_data_loader, valid_data_loader, criterion, optimizer, device, epochs, early_stopping_patience=5
):
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0

        # Training Loop
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(device), target.long().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation Step
        val_accuracy, val_precision, val_recall, val_f1, av_val_loss = validate_model(model, valid_data_loader,
                                                                                   criterion, device)
        print(f'Epoch: {epoch + 1}/{epochs}, '
              f'Training Loss: {running_loss/len(train_data_loader):.4f}, '
              f'Validation Loss: {av_val_loss:.4f}, '
              f'Accuracy: {val_accuracy:.4f}, '
              f'Precision: {val_precision:.4f}, '
              f'Recall: {val_recall:.4f}, '
              f'F1 Score: {val_f1:.4f}')

        # Check early stopping
        early_stopping(av_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

