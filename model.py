import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class LengthEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(LengthEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Molformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer.to(self.device)

    def forward(self, smiles):
        self.eval()
        with torch.no_grad():
            x = self.tokenizer(smiles, padding=True, return_tensors="pt").to(self.device)
            outputs = self.transformer(**x)
        return outputs.pooler_output


class MolPropPredictorMolFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = Molformer().transformer.config.hidden_size
        # self.transformer = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        # Freeze the transformer layers to only fine-tune the last layer
        # Freeze all transformer layers except the last one
        # for name, param in self.transformer.named_parameters():
        #     if "encoder.layer" in name:
        #         layer_num = int(name.split(".")[2])
        #         if layer_num < self.transformer.config.num_hidden_layers - 1:
        #             param.requires_grad = False
        #         else:
        #             param.requires_grad = True

        # Freeze the transformer layers to only fine-tune the last layer
        # for param in self.transformer.parameters():
        #     param.requires_grad = False

        # Define the final linear layer
        self.linear1 = nn.Linear(self.hidden_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.classifier = nn.Linear(in_features=256, out_features=2)
        self.relu = nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, molbert_embeddings):
        x = self.relu(self.linear1(self.relu(molbert_embeddings)))
        x = self.relu(self.linear2(x))
        x = self.classifier(x)
        return x


class MolPropPredictor(nn.Module):
    def __init__(self, mol_inp_size):
        super().__init__()
        self.linear1 = nn.Linear(mol_inp_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(in_features=128, out_features=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, baseline_model, noising_channel):
        super(HybridModel, self).__init__()
        self.noising_channel = noising_channel
        self.noising_channel.eval()
        self.baseline_model = baseline_model
        self.baseline_model.eval()
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            out = self.baseline_model(x)
            out = self.activation(out)
            predictions = self.noising_channel(out)
            return predictions


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


class Channel(nn.Module):
    """
    A PyTorch implementation of the Keras Channel layer suggested by the authors.

    Based on the paper:
    Goldberger & Ben-Reuven, Training deep neural-networks using a noise
    adaptation layer, ICLR 2017.
    https://openreview.net/forum?id=H12GRgcxg

    Arguments:
    - input_dim: int, the number of input dimensions (features).
    - output_dim: int, optional (default is the same as input_dim).
    - activation: activation function, default is softmax.
    - theta: optional, custom weights to initialize the channel matrix.
    """

    def __init__(self, input_dim, output_dim=None, activation=F.softmax, theta=None):
        super(Channel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.activation = activation

        # Initialize the channel matrix with custom weights (theta) if provided
        if theta is not None:
            self.channel_matrix = nn.Parameter(theta, requires_grad=True)
        else:
            self.channel_matrix = nn.Parameter(torch.randn(self.input_dim, self.output_dim), requires_grad=True)

    def forward(self, x):
        """
        Forward pass through the layer. Computes the dot product between the input
        and the channel matrix, applying the softmax to convert the channel matrix
        to a probability matrix.

        Arguments:
        - x: The output of the baseline classifier with shape (batch_size, input_dim).
        """
        # Convert channel_matrix to a stochastic matrix
        channel_matrix = self.activation(self.channel_matrix, dim=1)

        # Perform dot product: batch_size x input_dim with input_dim x output_dim
        return torch.matmul(x, channel_matrix)


class MNISTClassifier(nn.Module):
    def __init__(self, img_size=28*28, nhiddens1=500, nhiddens2=300, nb_classes=10, DROPOUT=0.5):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(img_size, nhiddens1)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(nhiddens1, nhiddens2)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.output = nn.Linear(nhiddens2, nb_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    import torch

    model = MolPropPredictorMolFormer()
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
    with torch.no_grad():
        outputs = model(smiles)
    print(outputs)