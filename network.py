import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Siren(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers,
                 out_features, outermost_linear=False,
                 qat=False):
        super().__init__()

        self.net = []
        if qat:
            self.net.append(torch.quantization.QuantStub())
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6/hidden_features) / 128,
                                              np.sqrt(6/hidden_features) / 128)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False))

        if qat:
            self.net.append(torch.quantization.DeQuantStub())

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True, is_first=False):
        super().__init__()
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features / 2,
                                             1 / self.in_features / 2)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / 128,
                                             np.sqrt(6/self.in_features) / 128)

    def forward(self, inputs):
        return torch.sin(128 * self.linear(inputs))


class NeuralFieldsNetwork(nn.Module):
    def __init__(self,
                 in_features: int, out_features: int,
                 hidden_features: int, n_hidden_layers: int,
                 input_embedding=None,
                 activation='ReLU',
                 output_activation=None,
                 use_qat=False):
        super(NeuralFieldsNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers

        activation = activation_mapper(activation)
        output_activation = activation_mapper(output_activation)

        self.net = []

        self.use_emb = input_embedding is not None
        self.use_qat = use_qat

        if self.use_emb:
            assert isinstance(input_embedding, Embedding)
            self.net.append(input_embedding)

        if use_qat:
            self.net.append(torch.quantization.QuantStub())

        self.net.extend([nn.Linear(input_embedding.get_output_size()
                                    if self.use_emb else in_features,
                                    hidden_features),
                          activation()])

        for i in range(self.n_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features),
                              activation()])

        self.net.extend([nn.Linear(hidden_features, out_features),
                          output_activation()])

        if use_qat:
            self.net.append(torch.quantization.DeQuantStub())

        self.net = nn.Sequential(*self.net)

    def forward(self, inputs):
        return self.net(inputs)


def activation_mapper(activation):
    if activation in [None, '']:
        return nn.Identity
    elif isinstance(activation, str):
        return getattr(torch.nn, activation)
    elif callable(activation):
        return activation
    else:
        raise ValueError()

