"""GC-SSM optimization, validation, and anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from gc_ssm.config import ExperimentConfig
from gc_ssm.data import CausalWindowDataset, split_training_windows
from gc_ssm.models import GCSSM, ModelOutput


GRADIENT_CLIP_NORM = 1.0
DATA_LOADER_WORKERS = 0


@dataclass(slots=True)
class TrainingResult:
    model: GCSSM
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    batch_size: int
    elapsed_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(
    n_features: int,
    model_window: int,
    config: ExperimentConfig,
    device: torch.device,
) -> GCSSM:
    return GCSSM(
        n_nodes=n_features,
        window_size=model_window,
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        node_emb_dim=config.node_emb_dim,
        forecast_steps=config.forecast_steps,
        random_feature_dim=config.random_feature_dim,
        random_feature_seed=config.random_feature_seed,
    ).to(device)


def compute_loss(
    targets: torch.Tensor,
    output: ModelOutput,
    lambda_forecast: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(output.reconstruction, targets)
    prediction_loss = targets.new_zeros(())
    for step, prediction in output.predictions.items():
        prediction_loss = prediction_loss + F.mse_loss(prediction, targets[:, step:])
    total = reconstruction_loss + lambda_forecast * prediction_loss
    return total, reconstruction_loss, prediction_loss


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_model(
    train_data: np.ndarray,
    n_features: int,
    model_window: int,
    config: ExperimentConfig,
    device: torch.device,
) -> TrainingResult:
    set_seed(config.seed)
    train_windows, validation_windows = split_training_windows(
        train_data, model_window, config.val_ratio
    )
    batch_size = min(config.batch_size, len(train_windows))
    pin_memory = device.type == "cuda"

    train_tensor = torch.from_numpy(train_windows).float()
    validation_tensor = torch.from_numpy(validation_windows).float()
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=pin_memory,
    )

    model = build_model(n_features, model_window, config, device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    best_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    print(
        f"Training GC-SSM: features={n_features}, window={model_window}, "
        f"train_windows={len(train_windows)}, validation_windows={len(validation_windows)}, "
        f"batch_size={batch_size}, device={device}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_total = 0.0
        train_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp, amp_dtype):
                total, _, _ = compute_loss(batch, model(batch), config.lambda_forecast)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(total.detach().cpu())
            train_batches += 1
        scheduler.step()

        model.eval()
        validation_total = 0.0
        validation_batches = 0
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                with _autocast_context(device, use_amp, amp_dtype):
                    total, _, _ = compute_loss(batch, model(batch), config.lambda_forecast)
                validation_total += float(total.detach().cpu())
                validation_batches += 1

        train_loss = train_total / train_batches
        validation_loss = validation_total / validation_batches
        if not math.isfinite(validation_loss):
            validation_loss = float("inf")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"Epoch {epoch:03d}/{config.epochs}: train={train_loss:.6f}, "
            f"validation={validation_loss:.6f}, "
            f"seconds={history[-1]['epoch_seconds']:.2f}",
            flush=True,
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            patience_count = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_epoch == 0 or not math.isfinite(best_loss):
        raise RuntimeError("Training did not produce a finite validation loss")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        batch_size=batch_size,
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def infer_scores(
    model: GCSSM,
    normalized_data: np.ndarray,
    model_window: int,
    lambda_forecast: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    dataset = CausalWindowDataset(normalized_data, model_window)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=device.type == "cuda",
    )
    scores = np.empty(len(dataset), dtype=np.float32)
    cursor = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        output = model(batch)
        reconstruction = (output.reconstruction[:, -1] - batch[:, -1]).pow(2).mean(dim=-1)
        prediction_score = torch.zeros_like(reconstruction)
        terms = 0
        for step, prediction in output.predictions.items():
            prediction_score = prediction_score + (
                prediction[:, -1] - batch[:, -1]
            ).pow(2).mean(dim=-1)
            terms += 1
        if terms:
            prediction_score = prediction_score / terms
        batch_scores = reconstruction + lambda_forecast * prediction_score
        values = batch_scores.float().cpu().numpy()
        scores[cursor : cursor + len(values)] = values
        cursor += len(values)
    return scores


def training_metadata(result: TrainingResult) -> dict[str, Any]:
    return {
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "epochs_completed": len(result.history),
        "batch_size": result.batch_size,
        "train_time_seconds": result.elapsed_seconds,
    }
