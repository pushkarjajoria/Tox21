from sklearn.metrics import roc_auc_score
import numpy as np


def multitask_auc(ground_truth, predicted):
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    n_tasks = ground_truth.shape[1]
    auc = []
    for i in range(n_tasks):
        ind = np.where(ground_truth[:, i] != 999)[0]
        auc.append(roc_auc_score(ground_truth[ind, i], predicted[ind, i]))
    return np.mean(auc)
