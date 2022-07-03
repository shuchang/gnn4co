import torch
from torch import nn

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