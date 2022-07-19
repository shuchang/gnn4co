import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    # 'selu': nn.SELU(),
    # 'softplus': nn.Softplus(),
    'identity': nn.Identity()}

def build_mlp(input_size: int, output_size: int, n_hidden_layers: int,
              hidden_size: int, activation='tanh', output_activation='identity'):
    """Builds a feed forward neural network"""
    activation = _str_to_activation[activation]
    output_activation = _str_to_activation[output_activation]

    layers = []
    in_size = input_size
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation)
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers)


class GAT(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x