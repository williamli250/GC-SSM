"""GC-SSM checkpoint serialization."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from gc_ssm.config import ExperimentConfig
from gc_ssm.data import PreparedSeries
from gc_ssm.models import GCSSM
from gc_ssm.models.block import mamba_available
from gc_ssm.training import TrainingResult, build_model, training_metadata


def save_checkpoint(
    path: str | Path,
    result: TrainingResult,
    config: ExperimentConfig,
    prepared: PreparedSeries,
    model_window: int,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "model_state": {
            name: value.detach().cpu() for name, value in result.model.state_dict().items()
        },
        "n_features": prepared.n_features,
        "feature_names": list(prepared.raw.feature_names),
        "model_window": model_window,
        "sliding_window": prepared.sliding_window,
        "scaler": {
            "center": prepared.scaler_center.tolist(),
            "scale": prepared.scaler_scale.tolist(),
        },
        "source_file": prepared.raw.path.name,
        "train_index": prepared.raw.train_index,
        "dropped_rows": prepared.raw.dropped_rows,
        "history": result.history,
        "training": training_metadata(result),
        "runtime": {
            "mamba_ssm_available": mamba_available(),
        },
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    required = {
        "config",
        "model_state",
        "n_features",
        "feature_names",
        "model_window",
        "sliding_window",
        "scaler",
        "source_file",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(missing)}")
    return payload


def restore_model(
    payload: dict[str, Any], device: torch.device
) -> tuple[GCSSM, ExperimentConfig]:
    config = ExperimentConfig.from_mapping(payload["config"])
    model = build_model(
        int(payload["n_features"]), int(payload["model_window"]), config, device
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, config
