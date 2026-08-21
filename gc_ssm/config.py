"""GC-SSM experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class ExperimentConfig:
    """Complete, serializable configuration for one GC-SSM experiment."""

    d_model: int = 32
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    num_blocks: int = 4
    dropout: float = 0.1
    node_emb_dim: int = 32
    random_feature_dim: int = 0
    random_feature_seed: int = 42
    forecast_steps: tuple[int, ...] = (10,)
    lambda_forecast: float = 1.0
    lr: float = 0.01
    weight_decay: float = 1e-3
    epochs: int = 100
    patience: int = 10
    val_ratio: float = 0.15
    batch_size: int = 128
    seed: int = 2024
    device: str = "auto"
    model_window: int | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        normalized = dict(values)
        if "forecast_steps" in normalized:
            raw_steps = normalized["forecast_steps"]
            if isinstance(raw_steps, int):
                raw_steps = [raw_steps]
            normalized["forecast_steps"] = tuple(int(step) for step in raw_steps)
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        positive_ints = {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "num_blocks": self.num_blocks,
            "node_emb_dim": self.node_emb_dim,
            "epochs": self.epochs,
            "patience": self.patience,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, received {value}")
        if not self.forecast_steps or any(step <= 0 for step in self.forecast_steps):
            raise ValueError("forecast_steps must contain positive integers")
        if len(set(self.forecast_steps)) != len(self.forecast_steps):
            raise ValueError("forecast_steps must not contain duplicates")
        if not isinstance(self.random_feature_dim, int) or isinstance(
            self.random_feature_dim, bool
        ):
            raise ValueError("random_feature_dim must be a non-negative integer")
        if self.random_feature_dim < 0:
            raise ValueError("random_feature_dim must be a non-negative integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1)")
        if self.lambda_forecast < 0.0:
            raise ValueError("lambda_forecast must be non-negative")
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("lr must be positive and weight_decay must be non-negative")
        minimum_horizon_window = max(self.forecast_steps) + 1
        if self.model_window is not None and self.model_window < minimum_horizon_window:
            raise ValueError(
                f"model_window must be at least {minimum_horizon_window} for the configured forecast steps"
            )
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise ValueError("batch_size must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["forecast_steps"] = list(self.forecast_steps)
        return result


def load_config(overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Apply explicit command-line overrides to the built-in defaults."""
    values: dict[str, Any] = ExperimentConfig().to_dict()
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig.from_mapping(values)
