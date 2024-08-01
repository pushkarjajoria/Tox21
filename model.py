from torch import nn


class MolPropPredictor(nn.Module):
    def __init__(self, mol_inp_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=mol_inp_size, hidden_size=128, num_layers=2, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(in_features=128, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
