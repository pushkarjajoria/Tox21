import pickle

import numpy as np
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import get_tokens
from sklearn.model_selection import train_test_split
from openchem.data.utils import save_smiles_property_file


data = read_smiles_property_file('../OpenChem/benchmark_datasets/tox21/tox21.csv',
                                 cols_to_read=[13] + list(range(0,12)))
smiles = data[0]
labels = np.array(data[1:])

labels[np.where(labels=='')] = '9'
labels = labels.astype(int)
# labels = labels.reshape(labels.shape[0], 1)
labels = labels.T

tokens, _, _ = get_tokens(smiles)
tokens = tokens + ' '
with open("benchmark_datasets/tox21/tokens.txt", 'wb') as f:
    pickle.dump(tokens, f)
X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                    random_state=42)

save_smiles_property_file('./benchmark_datasets/tox21/train.smi', X_train, y_train)
save_smiles_property_file('./benchmark_datasets/tox21/test.smi', X_test, y_test)
