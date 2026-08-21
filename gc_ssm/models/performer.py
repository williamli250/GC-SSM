"""FAVOR+ graph structure learner."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def orthogonal_random_matrix(rows: int, columns: int, seed: int) -> torch.Tensor:
    """Create the fixed orthogonal random features used by Performer."""
    if rows <= 0 or columns <= 0:
        raise ValueError("Random projection dimensions must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    blocks = []
    for _ in range((rows + columns - 1) // columns):
        raw = torch.randn(columns, columns, generator=generator)
        q, _ = torch.linalg.qr(raw)
        blocks.append(q.t())
    projection = torch.cat(blocks, dim=0)[:rows]
    norms = torch.randn(rows, columns, generator=generator).norm(dim=1)
    return projection * norms.unsqueeze(-1)


def favor_feature_map(x: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """Apply the positive orthogonal random-feature map."""
    projected = x @ projection.t()
    squared_norm = 0.5 * (x * x).sum(dim=-1, keepdim=True)
    stabilizer = projected.amax(dim=-1, keepdim=True)
    return torch.exp(projected - squared_norm - stabilizer) / math.sqrt(projection.shape[0])


class PerformerGraph(nn.Module):
    """Produce asymmetric query/key FAVOR+ features for every sensor."""

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        d_emb: int,
        window_size: int,
        m_features: int,
        random_feature_seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.d_emb = d_emb
        self.m_features = m_features

        self.node_emb = nn.Parameter(torch.empty(n_nodes, d_emb))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_proj = nn.Linear(window_size, 1, bias=False)
        # Keep the published parameter names aligned with the research implementation.
        self.linear_Q = nn.Linear(d_model + d_emb, d_model)
        self.linear_K = nn.Linear(d_model + d_emb, d_model)
        self.tau = nn.Parameter(torch.tensor(1.0))
        projection = orthogonal_random_matrix(m_features, d_model, random_feature_seed)
        self.register_buffer("rand_proj", projection)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 4 or h.shape[2] != self.n_nodes or h.shape[3] != self.d_model:
            raise ValueError("PerformerGraph received an incompatible feature tensor")
        batch_size = h.shape[0]
        pooled = self.time_proj(h.permute(0, 2, 3, 1)).squeeze(-1)
        node_ids = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([pooled, node_ids], dim=-1)

        scale = torch.rsqrt(self.tau.clamp(min=0.01))
        query = F.normalize(self.linear_Q(features), p=2, dim=-1) * scale
        key = F.normalize(self.linear_K(features), p=2, dim=-1) * scale
        return favor_feature_map(query, self.rand_proj), favor_feature_map(key, self.rand_proj)

    @staticmethod
    def materialize_adjacency(
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        *,
        remove_self_loops: bool = True,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Materialize a dense adjacency for inspection only, never for model execution."""
        adjacency = torch.matmul(phi_q, phi_k.transpose(-1, -2))
        if remove_self_loops:
            n_nodes = adjacency.shape[-1]
            mask = torch.eye(n_nodes, dtype=torch.bool, device=adjacency.device)
            adjacency = adjacency.masked_fill(mask.unsqueeze(0), 0.0)
        denominator = adjacency.sum(dim=-1, keepdim=True).clamp_min(eps)
        return adjacency / denominator
