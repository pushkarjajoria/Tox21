import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def test(test_loader, model):
    all_preds = []
    all_labels = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in test_loader:
            x = batch['fingerprint'].float().to(device)
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
        x = batch['fingerprint'].float().to(device)
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
