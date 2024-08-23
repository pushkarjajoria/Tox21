from torch import nn


class LengthEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(LengthEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)


class MolPropPredictor(nn.Module):
    def __init__(self, mol_inp_size):
        super().__init__()
        self.linear1 = nn.Linear(mol_inp_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x
