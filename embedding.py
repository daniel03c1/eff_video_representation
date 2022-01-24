import torch
import torch.nn as nn


class Embedding(nn.Module):
    def forward(self, inputs):
        raise NotImplemented()

    def get_output_size(self):
        raise NotImplemented()


class RFF(Embedding):
    def __init__(self, in_features, emb_size, emb_sigma):
        super().__init__()
        self.B = nn.Parameter(
            torch.normal(0, emb_sigma, (in_features, emb_size)),
            requires_grad=False)
        self.output_size = emb_size * 2

    def forward(self, inputs):
        outs = inputs @ self.B.to(inputs.device)
        return torch.cat([torch.sin(outs), torch.cos(outs)], -1)

    def get_output_size(self):
        return self.output_size 


class PosEncoding(Embedding):
    def __init__(self, in_features, n_freqs, include_inputs=False):
        super().__init__()
        self.in_features = in_features

        if isinstance(n_freqs, int):
            n_freqs = [n_freqs for _ in range(in_features)]
        self.n_freqs = n_freqs

        eye = torch.eye(in_features)
        self.freq_mat = nn.Parameter(
            torch.cat([torch.stack([eye[i] * (2**j)
                                    for j in range(self.n_freqs[i])], -1)
                       for i in range(in_features)], -1),
            requires_grad=False)

        self.include_inputs = include_inputs
        self.output_size = in_features * include_inputs + 2 * sum(self.n_freqs)

    def forward(self, inputs):
        outs = []
        if self.include_inputs:
            outs.append(inputs)
        mapped = inputs @ self.freq_mat
        outs.append(torch.cos(mapped))
        outs.append(torch.sin(mapped))

        return torch.cat(outs, -1)

    def get_output_size(self):
        return self.output_size

