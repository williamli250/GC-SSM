"""Graph-conditioned selective state-space block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    import selective_scan_cuda as _selective_scan_cuda
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    MAMBA_AVAILABLE = True
except (ImportError, OSError):
    _selective_scan_cuda = None
    selective_scan_fn = None
    MAMBA_AVAILABLE = False


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, dimension: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inverse_rms * self.weight


class GCSSMBlock(nn.Module):
    """One graph-conditioned selective SSM block."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        d_emb: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.d_emb = d_emb

        inner = self.d_inner
        self.norm = RMSNorm(d_model)
        self.linear_expand = nn.Linear(d_model, inner)
        self.linear_gate = nn.Linear(d_model, inner)
        self.conv1d = nn.Conv1d(
            inner,
            inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=inner,
            bias=True,
        )
        self.linear_msg = nn.Linear(inner + d_emb, inner, bias=False)
        self.linear_delta = nn.Linear(2 * inner, inner)
        self.linear_B = nn.Linear(2 * inner, d_state, bias=False)
        self.linear_C = nn.Linear(2 * inner, d_state, bias=False)

        a_init = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(inner, -1)
        )
        self.A_log = nn.Parameter(a_init)
        self.D_skip = nn.Parameter(torch.ones(inner))
        self.linear_out = nn.Linear(inner, d_model)
        self.drop = nn.Dropout(dropout)

    def graph_aggregate(
        self,
        values: torch.Tensor,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate messages in linear node complexity with self-loop suppression."""
        batch_size, time_steps, n_nodes, _ = values.shape
        expanded_ids = node_emb.unsqueeze(0).unsqueeze(0).expand(
            batch_size, time_steps, -1, -1
        )
        messages = self.linear_msg(torch.cat([values, expanded_ids], dim=-1))

        accumulator = torch.einsum("bnk,btne->btke", phi_k, messages)
        numerator = torch.einsum("bnk,btke->btne", phi_q, accumulator)
        key_sum = phi_k.sum(dim=1)
        denominator = torch.einsum("bnk,bk->bn", phi_q, key_sum)

        self_kernel = (phi_q * phi_k).sum(dim=-1)
        numerator = numerator - self_kernel[:, None, :, None] * messages
        denominator = denominator - self_kernel
        denominator = denominator[:, None, :, None].clamp_min(1e-6)
        return numerator / denominator

    def _ssm_scan_cuda(
        self,
        values: torch.Tensor,
        delta_raw: torch.Tensor,
        b_values: torch.Tensor,
        c_values: torch.Tensor,
    ) -> torch.Tensor:
        if selective_scan_fn is None:
            raise RuntimeError("The Mamba selective scan kernel is not available")
        batch_size, time_steps, n_nodes, inner = values.shape

        def to_kernel(tensor: torch.Tensor, add_group: bool = False) -> torch.Tensor:
            result = (
                tensor.permute(0, 2, 3, 1)
                .reshape(batch_size * n_nodes, -1, time_steps)
                .contiguous()
            )
            return result.unsqueeze(1) if add_group else result

        output = selective_scan_fn(
            to_kernel(values),
            to_kernel(delta_raw),
            -torch.exp(self.A_log),
            to_kernel(b_values, add_group=True),
            to_kernel(c_values, add_group=True),
            self.D_skip,
            delta_softplus=True,
            return_last_state=False,
        )
        return output.reshape(batch_size, n_nodes, inner, time_steps).permute(0, 3, 1, 2)

    def _ssm_scan_fallback(
        self,
        values: torch.Tensor,
        delta_raw: torch.Tensor,
        b_values: torch.Tensor,
        c_values: torch.Tensor,
    ) -> torch.Tensor:
        """Autograd-safe pure PyTorch selective scan."""
        batch_size, time_steps, n_nodes, inner = values.shape
        a_values = -torch.exp(self.A_log)
        state = values.new_zeros(batch_size, n_nodes, inner, self.d_state)
        outputs = []
        for index in range(time_steps):
            delta = F.softplus(delta_raw[:, index]).unsqueeze(-1)
            a_bar = torch.exp(delta * a_values)
            b_bar = delta * b_values[:, index].unsqueeze(2)
            state = a_bar * state + values[:, index].unsqueeze(-1) * b_bar
            outputs.append(
                torch.einsum("bnes,bns->bne", state, c_values[:, index])
                + self.D_skip * values[:, index]
            )
        return torch.stack(outputs, dim=1)

    def selective_scan(
        self,
        values: torch.Tensor,
        delta_raw: torch.Tensor,
        b_values: torch.Tensor,
        c_values: torch.Tensor,
    ) -> torch.Tensor:
        if MAMBA_AVAILABLE and values.is_cuda:
            return self._ssm_scan_cuda(values, delta_raw, b_values, c_values)
        return self._ssm_scan_fallback(values, delta_raw, b_values, c_values)

    def forward(
        self,
        x: torch.Tensor,
        graph_features: tuple[torch.Tensor, torch.Tensor],
        node_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, n_nodes, _ = x.shape
        normalized = self.norm(x)
        expanded = self.linear_expand(normalized)
        gate = F.silu(self.linear_gate(normalized))

        convolved = expanded.permute(0, 2, 3, 1).reshape(
            batch_size * n_nodes, self.d_inner, time_steps
        )
        convolved = self.conv1d(convolved)[:, :, :time_steps]
        values = F.silu(
            convolved.reshape(batch_size, n_nodes, self.d_inner, time_steps).permute(
                0, 3, 1, 2
            )
        )

        phi_q, phi_k = graph_features
        neighbors = self.graph_aggregate(values, phi_q, phi_k, node_emb)
        conditioned = torch.cat([values, neighbors], dim=-1)
        delta_raw = self.linear_delta(conditioned)
        b_values = self.linear_B(conditioned)
        c_values = self.linear_C(conditioned)

        scanned = self.selective_scan(values, delta_raw, b_values, c_values)
        return self.drop(self.linear_out(scanned * gate)) + x


def mamba_available() -> bool:
    return MAMBA_AVAILABLE
