import pickle

import torch
from openchem.data.smiles_data_layer import SmilesDataset
from torch.utils.data import DataLoader

from model import MolPropPredictor

with open("./benchmark_datasets/tox21/tokens.txt", 'rb') as f:
    tokens = pickle.load(f)

train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi',
                              delimiter=',', cols_to_read=list(range(2)),
                              tokens=tokens, augment=True)
test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                            delimiter=',', cols_to_read=list(range(2)),
                            tokens=tokens)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

inp_size = train_dataset.data.shape[0]
model = MolPropPredictor(inp_size)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

epochs = 100

for epoch in range(epochs):
    model.train()
    for batch in train_data_loader:
        optim.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optim.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')


