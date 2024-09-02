import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LengthEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(LengthEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)


class MolPropPredictor(nn.Module):
    def __init__(self, mol_inp_size):
        super().__init__()
        self.linear1 = nn.Linear(mol_inp_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(in_features=256, out_features=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class NoiseLayer(nn.Module):
    def __init__(self, theta, k):
        super(NoiseLayer, self).__init__()
        self.theta = nn.Linear(k, k, bias=False)
        self.theta.weight.data = nn.Parameter(theta)
        self.eye = torch.Tensor(np.eye(k))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        theta = self.eye.to(x.device).detach()
        theta = self.theta(theta)
        theta = self.softmax(theta)
        out = torch.matmul(x, theta)
        return out