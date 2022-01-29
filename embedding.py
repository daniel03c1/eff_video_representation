import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_input_grid


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

        if isinstance(n_freqs, (int, float)):
            n_freqs = [n_freqs for _ in range(in_features)]
        self.n_freqs = torch.tensor(n_freqs).int()

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


class MultiHashEncoding(Embedding):
    def __init__(self, video, embedding_dim, grid_size=8, n_levels=1):
        super().__init__()
        T, _, H, W = video.size()
        self.volume = torch.tensor([T, H, W])
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.n_levels = n_levels

        self.n_dim = 3

        self.grid_shape = [
            torch.ceil(self.volume / self.grid_size / (2**i)) + 1
            for i in range(n_levels)]
        self.offset = torch.tensor([[int(j) for j in f'{i:03b}']
                                    for i in range(2**self.n_dim)])

        embeddings = []
        for i in range(self.n_levels):
            grid = torch.flip(make_input_grid(*self.grid_shape[i],
                                              minvalue=-1).unsqueeze(0),
                              (-1,))
            colors = F.grid_sample(video.permute(1, 0, 2, 3).unsqueeze(0),
                                   grid, padding_mode='reflection',
                                   align_corners=True).squeeze(0)
            colors = colors.permute(1, 2, 3, 0) # T, H, W, C
            colors = colors - colors.reshape(-1, 3).mean(dim=(-2,))
            mat = torch.pca_lowrank(colors.reshape(-1, 3), self.embedding_dim, center=False)[-1]
            colors = colors @ mat
            # colors = torch.rand_like(colors) * 2e-4 - 1e-4
            embeddings.append(nn.Parameter(colors))
        self.embeddings = nn.ParameterList(embeddings)

        self.output_size = embedding_dim * n_levels

    def forward(self, inputs):
        outputs = []
        for i in range(self.n_levels):
            # [0, 1] to [0, size-1]
            coords = inputs * (self.grid_shape[i].to(inputs.device) - 1)

            out = torch.floor(self.offset.to(coords.device) + coords[..., None, :])
            out = torch.clamp(out, min=torch.zeros(self.n_dim, device=out.device),
                              max=self.grid_shape[i].to(out.device) - 1)

            values = self.embeddings[i][out[..., 0].long(),
                                        out[..., 1].long(),
                                        out[..., 2].long()]

            weights = 1 - torch.abs(out - coords[..., None, :])
            # bilinear interpolation (identity)
            weights = weights.prod(-1, keepdim=True)

            out = torch.sum(weights * values, -2)

            outputs.append(out)

        return torch.cat(outputs, -1)

    def get_output_size(self):
        return self.output_size

