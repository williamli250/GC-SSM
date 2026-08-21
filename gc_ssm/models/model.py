"""GC-SSM architecture and task-specific output heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gc_ssm.models.block import GCSSMBlock
from gc_ssm.models.embedding import PerSensorEmbedding
from gc_ssm.models.performer import PerformerGraph


class TaskHead(nn.Module):
    """LayerNorm, linear projection, GELU, and scalar projection."""

    def __init__(self, d_model: int, d_hidden: int | None = None) -> None:
        super().__init__()
        hidden = d_hidden or d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class ModelOutput:
    """Outputs produced by a GC-SSM forward pass."""

    reconstruction: torch.Tensor
    features: torch.Tensor
    predictions: dict[int, torch.Tensor]


class GCSSM(nn.Module):
    """Graph-Conditioned Selective State Space Model."""

    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        d_model: int = 32,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        num_blocks: int = 4,
        dropout: float = 0.1,
        node_emb_dim: int = 32,
        forecast_steps: tuple[int, ...] = (10,),
        random_feature_dim: int = 0,
        random_feature_seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.d_model = d_model
        self.forecast_steps = tuple(forecast_steps)

        self.embedding = PerSensorEmbedding(n_nodes, d_model)
        feature_dim = random_feature_dim if random_feature_dim > 0 else 2 * d_model
        self.graph = PerformerGraph(
            n_nodes=n_nodes,
            d_model=d_model,
            d_emb=node_emb_dim,
            window_size=window_size,
            m_features=feature_dim,
            random_feature_seed=random_feature_seed,
        )
        self.blocks = nn.ModuleList(
            [
                GCSSMBlock(d_model, d_state, d_conv, expand, node_emb_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.recon_head = TaskHead(d_model)
        self.forecast_heads = nn.ModuleDict(
            {f"step_{step}": TaskHead(d_model) for step in self.forecast_steps}
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        features = self.embedding(x)
        graph_features = self.graph(features)
        for block in self.blocks:
            features = block(features, graph_features, self.graph.node_emb)
        reconstruction = self.recon_head(features)
        predictions = {
            step: self.forecast_heads[f"step_{step}"](features[:, :-step])
            for step in self.forecast_steps
            if features.shape[1] > step
        }
        return ModelOutput(reconstruction, features, predictions)
