"""TSB-AD metric integration and benchmark aggregation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


METRIC_KEYS = (
    "VUS-PR",
    "VUS-ROC",
    "AUC-PR",
    "AUC-ROC",
)


def evaluate_scores(
    scores: np.ndarray, labels: np.ndarray, sliding_window: int
) -> dict[str, float]:
    """Calculate the four threshold-independent TSB-AD metrics."""
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("scores and labels must be one-dimensional arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores contain non-finite values")
    try:
        from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve
    except ImportError as exc:
        raise ImportError(
            "TSB-AD is required for benchmark metrics. Install TSB-AD."
        ) from exc

    evaluator = basic_metricor()
    auc_roc = evaluator.metric_ROC(labels, scores)
    auc_pr = evaluator.metric_PR(labels, scores)
    *_, vus_roc, vus_pr = generate_curve(
        labels.astype(int), scores, sliding_window, "opt", 250
    )
    metrics = {
        "VUS-PR": float(vus_pr),
        "VUS-ROC": float(vus_roc),
        "AUC-PR": float(auc_pr),
        "AUC-ROC": float(auc_roc),
    }
    non_finite = [key for key, value in metrics.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(f"TSB-AD returned non-finite metrics: {', '.join(non_finite)}")
    return metrics


def summarize_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(list(rows))
    summary: dict[str, Any] = {"n_files": len(frame)}
    for key in METRIC_KEYS:
        if key not in frame:
            continue
        values = pd.to_numeric(frame[key], errors="coerce").dropna()
        if values.empty:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(values.median())
        summary[f"{key}_std"] = float(values.std()) if len(values) > 1 else 0.0
    return summary
