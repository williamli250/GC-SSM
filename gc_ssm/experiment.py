"""GC-SSM training, evaluation, and benchmark workflows."""

from __future__ import annotations

import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch

from gc_ssm.checkpoint import load_checkpoint, restore_model, save_checkpoint
from gc_ssm.config import ExperimentConfig
from gc_ssm.data import (
    apply_saved_scaler,
    load_file_list,
    load_raw_series,
    prepare_series,
    resolve_model_window,
)
from gc_ssm.evaluation import evaluate_scores, summarize_metrics
from gc_ssm.training import (
    infer_scores,
    resolve_device,
    train_model,
    training_metadata,
)


EXPECTED_EVALUATION_FILES = 180


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def train_experiment(
    data_path: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw_series(data_path)
    prepared = prepare_series(raw)
    model_window = resolve_model_window(
        prepared.sliding_window,
        config.forecast_steps,
        config.model_window,
    )
    if len(prepared.train_segment) < model_window:
        raise ValueError("The training prefix is shorter than the resolved model window")

    print(
        f"Preparing {raw.path.name}: rows={len(raw.features)}, "
        f"features={prepared.n_features}, train_prefix={raw.train_index}, "
        f"sliding_window={prepared.sliding_window}, model_window={model_window}",
        flush=True,
    )

    device = resolve_device(config.device)
    result = train_model(
        prepared.train_segment,
        prepared.n_features,
        model_window,
        config,
        device,
    )
    checkpoint_path = save_checkpoint(
        output / "checkpoint.pt", result, config, prepared, model_window
    )
    _write_json(output / "resolved_config.json", config.to_dict())
    _write_json(
        output / "history.json",
        {"history": result.history, **training_metadata(result)},
    )
    return checkpoint_path


def evaluate_experiment(
    data_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device_override: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested_device = device_override or "auto"
    device = resolve_device(requested_device)
    payload = load_checkpoint(checkpoint_path, device)
    model, config = restore_model(payload, device)
    raw = load_raw_series(data_path)

    if raw.path.name != payload["source_file"]:
        raise ValueError(
            f"Checkpoint was trained for {payload['source_file']!r}, not {raw.path.name!r}"
        )
    if raw.features.shape[1] != int(payload["n_features"]):
        raise ValueError("Dataset feature count does not match the checkpoint")
    if list(raw.feature_names) != list(payload["feature_names"]):
        raise ValueError("Dataset feature names or order do not match the checkpoint")

    normalized = apply_saved_scaler(
        raw.features,
        payload["scaler"]["center"],
        payload["scaler"]["scale"],
    )
    scores = infer_scores(
        model,
        normalized,
        int(payload["model_window"]),
        config.lambda_forecast,
        config.batch_size,
        device,
    )
    metrics = evaluate_scores(scores, raw.labels, int(payload["sliding_window"]))
    result: dict[str, Any] = {
        "file": raw.path.name,
        **metrics,
    }
    np.save(output / "anomaly_scores.npy", scores)
    np.save(output / "labels.npy", raw.labels)
    _write_json(output / "metrics.json", result)
    _write_json(output / "resolved_config.json", config.to_dict())
    if "history" in payload:
        _write_json(
            output / "history.json",
            {"history": payload["history"], **payload.get("training", {})},
        )
    return result


def run_experiment(
    data_path: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    checkpoint = train_experiment(data_path, config, output_dir)
    return evaluate_experiment(
        data_path,
        checkpoint,
        output_dir,
        device_override=config.device,
    )


def benchmark_experiment(
    dataset_dir: str | Path,
    file_list_path: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    names = load_file_list(file_list_path)
    if len(names) != EXPECTED_EVALUATION_FILES:
        raise ValueError(
            "The TSB-AD-M evaluation file list must contain exactly "
            f"{EXPECTED_EVALUATION_FILES} files; received {len(names)}"
        )
    missing = [name for name in names if not (dataset_root / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} files from the file list are missing under {dataset_root}: {preview}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, name in enumerate(names, start=1):
        csv_path = dataset_root / name
        file_output = output / csv_path.stem
        print(f"[{index}/{len(names)}] run: {name}", flush=True)
        try:
            rows.append(run_experiment(csv_path, config, file_output))
        except Exception as exc:
            errors.append(
                {
                    "file": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_file_metrics.csv", index=False)
    summary = {
        "seed": config.seed,
        "file_list": str(Path(file_list_path)),
        "dataset_dir": str(dataset_root),
        "n_expected": len(names),
        "n_completed": len(rows),
        "n_errors": len(errors),
        **summarize_metrics(rows),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "errors.json", errors)
    _write_json(output / "resolved_config.json", config.to_dict())
    return summary
