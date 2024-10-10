from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def multitask_auc(ground_truth, predicted):
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    n_tasks = ground_truth.shape[1]
    auc = []
    for i in range(n_tasks):
        ind = np.where(ground_truth[:, i] != 999)[0]
        auc.append(roc_auc_score(ground_truth[ind, i], predicted[ind, i]))
    return np.mean(auc)


def get_all_pred_and_labels(model, train_data_loader, fingerprint=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_labels = []
    all_preds = []
    for batch in train_data_loader:
        if fingerprint:
            x = batch['x'].float().to(device)
        else:
            x = batch['x']
        labels = batch['label'].long().to(device)  # Labels should be of type long for CrossEntropyLoss
        output = model(x)
        pred = torch.argmax(output, dim=1).cpu()  # Apply threshold for binary classification
        all_labels.extend(labels.cpu().numpy())  # Store true labels
        all_preds.extend(pred.numpy())  # Store predicted labels

    return all_preds, all_labels


def get_all_pred_and_labels_mnist(model, train_data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_labels = []
    all_preds = []

    model.eval()  # Set model to evaluation mode (disable dropout, batch norm, etc.)

    with torch.no_grad():  # Disable gradient computation
        for batch in train_data_loader:
            x = batch[0].float().to(device)  # Use batch[0] for input images (MNIST)
            labels = batch[1].long().to(device)  # Use batch[1] for labels (MNIST)
            output = model(x)
            preds = torch.argmax(output, dim=1).cpu()  # Get predicted class index
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.numpy())  # Store predicted labels

    return all_preds, all_labels
