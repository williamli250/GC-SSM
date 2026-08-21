"""Per-sensor scalar embeddings."""

import torch
from torch import nn


class PerSensorEmbedding(nn.Module):
    """Map each scalar sensor value with an independent affine projection."""

    def __init__(self, n_nodes: int, d_model: int) -> None:
        super().__init__()
        if n_nodes < 2:
            raise ValueError("GC-SSM requires at least two sensor channels")
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(n_nodes, d_model))
        self.bias = nn.Parameter(torch.zeros(n_nodes, d_model))
        nn.init.uniform_(self.weight, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.n_nodes:
            raise ValueError(
                f"Expected input shape (batch, time, {self.n_nodes}), received {tuple(x.shape)}"
            )
        return x.unsqueeze(-1) * self.weight + self.bias
