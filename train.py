import pickle

import torch
from openchem.data.smiles_data_layer import SmilesDataset
from torch.utils.data import DataLoader
from model import MolPropPredictor
from tqdm import tqdm

with open("./benchmark_datasets/tox21/tokens.txt", 'rb') as f:
    tokens = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running the model on {device}")
print(torch.version.cuda)
train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi',
                              delimiter=',', cols_to_read=list(range(2)),
                              tokens=tokens, augment=True)
test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                            delimiter=',', cols_to_read=list(range(2)),
                            tokens=tokens)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

inp_size = train_dataset.data.shape[1]
model = MolPropPredictor(inp_size).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss().to(device)

epochs = 100

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_data_loader:
        x = batch['tokenized_smiles'].float().to(device)
        labels = batch['labels'].float().to(device)
        lengths = batch['length']
        optim.zero_grad()
        output = model(x, lengths)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')


