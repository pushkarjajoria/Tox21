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


def get_all_pred_and_labels(model, train_data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_labels = []
    all_preds = []
    for batch in train_data_loader:
        x = batch['fingerprint'].float().to(device)
        labels = batch['label'].long().to(device)  # Labels should be of type long for CrossEntropyLoss
        output = model(x)
        pred = torch.argmax(output, dim=1).cpu()  # Apply threshold for binary classification
        all_labels += labels
        all_preds += pred
    return all_preds, all_labels

