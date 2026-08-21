"""TSB-AD-M data loading, normalization, windowing, and validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset


TRAIN_INDEX_PATTERN = re.compile(r"_tr_(\d+)_")
MODEL_WINDOW_MIN = 11
MODEL_WINDOW_MAX = 128


@dataclass(slots=True)
class RawSeries:
    path: Path
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    train_index: int
    dropped_rows: int


@dataclass(slots=True)
class PreparedSeries:
    raw: RawSeries
    normalized: np.ndarray
    train_segment: np.ndarray
    sliding_window: int
    scaler_center: np.ndarray
    scaler_scale: np.ndarray

    @property
    def n_features(self) -> int:
        return self.normalized.shape[1]


def parse_train_index(filename: str) -> int:
    """Parse the TSB-AD training-prefix length from a benchmark filename."""
    match = TRAIN_INDEX_PATTERN.search(Path(filename).name)
    if match is None:
        raise ValueError(
            f"Cannot parse a training prefix from {filename!r}; expected '_tr_<N>_'"
        )
    value = int(match.group(1))
    if value <= 0:
        raise ValueError("The training prefix length must be positive")
    return value


def load_raw_series(path: str | Path) -> RawSeries:
    """Read and validate one multivariate TSB-AD-M CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "Label" not in frame.columns:
        raise ValueError(f"Dataset must contain a 'Label' column: {csv_path}")
    if frame.columns.tolist().count("Label") != 1:
        raise ValueError("Dataset must contain exactly one 'Label' column")

    original_rows = len(frame)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("Dataset is empty after dropping rows with missing values")
    dropped_rows = original_rows - len(frame)

    feature_frame = frame.drop(columns=["Label"])
    if feature_frame.shape[1] < 2:
        raise ValueError("GC-SSM requires at least two feature columns")
    try:
        features = feature_frame.to_numpy(dtype=np.float64)
        label_values = pd.to_numeric(frame["Label"], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature and Label values must be numeric") from exc
    if not np.isfinite(features).all() or not np.isfinite(label_values).all():
        raise ValueError("Dataset contains non-finite values")
    if not np.all(np.isin(label_values, [0, 1])):
        raise ValueError("Label values must be binary integers 0 or 1")

    train_index = parse_train_index(csv_path.name)
    if train_index >= len(features):
        raise ValueError(
            f"Training prefix {train_index} must be shorter than the series length {len(features)}"
        )
    return RawSeries(
        path=csv_path,
        features=features,
        labels=label_values.astype(np.int64),
        feature_names=tuple(str(name) for name in feature_frame.columns),
        train_index=train_index,
        dropped_rows=dropped_rows,
    )


def detect_sliding_window(features: np.ndarray) -> int:
    """Use the official TSB-AD ACF heuristic on the full first channel."""
    try:
        from TSB_AD.utils.slidingWindows import find_length_rank
    except ImportError as exc:
        raise ImportError(
            "TSB-AD is required for automatic window detection. Install TSB-AD."
        ) from exc
    window = int(find_length_rank(features[:, 0].reshape(-1, 1), rank=1))
    if window <= 0:
        raise ValueError(f"TSB-AD returned an invalid sliding window: {window}")
    return window


def prepare_series(raw: RawSeries) -> PreparedSeries:
    """Fit RobustScaler on the training prefix and transform the full series."""
    training = raw.features[: raw.train_index]
    scaler = RobustScaler().fit(training)
    normalized = scaler.transform(raw.features).astype(np.float32)
    return PreparedSeries(
        raw=raw,
        normalized=normalized,
        train_segment=normalized[: raw.train_index],
        sliding_window=detect_sliding_window(raw.features),
        scaler_center=np.asarray(scaler.center_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
    )


def apply_saved_scaler(
    features: np.ndarray,
    center: Iterable[float],
    scale: Iterable[float],
) -> np.ndarray:
    """Apply checkpointed RobustScaler statistics without fitting again."""
    center_array = np.asarray(center, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if center_array.shape != (features.shape[1],) or scale_array.shape != (features.shape[1],):
        raise ValueError("Checkpoint scaler statistics do not match the dataset feature count")
    safe_scale = np.where(scale_array == 0.0, 1.0, scale_array)
    return ((features - center_array) / safe_scale).astype(np.float32)


def resolve_model_window(
    sliding_window: int,
    forecast_steps: tuple[int, ...],
    forced_window: int | None,
) -> int:
    """Resolve the model window using the published TSB-AD-M clamp."""
    lower_bound = max(MODEL_WINDOW_MIN, max(forecast_steps) + 1)
    if lower_bound > MODEL_WINDOW_MAX:
        raise ValueError(
            "forecast_steps require a model window larger than the supported "
            f"maximum of {MODEL_WINDOW_MAX}"
        )
    model_window = (
        forced_window
        if forced_window is not None
        else min(sliding_window, MODEL_WINDOW_MAX)
    )
    model_window = max(model_window, lower_bound)
    if forced_window is not None and forced_window > MODEL_WINDOW_MAX:
        raise ValueError(f"model_window cannot exceed {MODEL_WINDOW_MAX}")
    return model_window


def make_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Create unpadded overlapping training windows."""
    if data.ndim != 2:
        raise ValueError("Expected a two-dimensional time-by-feature array")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(data) < window_size:
        raise ValueError(
            f"Series length {len(data)} is shorter than window_size={window_size}"
        )
    indices = np.arange(window_size)[None, :] + np.arange(
        len(data) - window_size + 1
    )[:, None]
    return data[indices]


def split_training_windows(
    data: np.ndarray, window_size: int, val_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split overlapping windows chronologically into train and validation sets."""
    windows = make_windows(data, window_size)
    split = int(len(windows) * (1.0 - val_ratio))
    if split <= 0 or split >= len(windows):
        raise ValueError(
            "The training prefix is too short to produce non-empty train and validation sets"
        )
    return windows[:split], windows[split:]


class CausalWindowDataset(Dataset[torch.Tensor]):
    """Generate causal inference windows lazily to avoid materializing the full tensor."""

    def __init__(self, data: np.ndarray, window_size: int) -> None:
        if data.ndim != 2 or len(data) == 0:
            raise ValueError("A non-empty two-dimensional array is required")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.data = data
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = max(0, index - self.window_size + 1)
        window = self.data[start : index + 1]
        missing = self.window_size - len(window)
        if missing:
            window = np.concatenate([np.repeat(self.data[:1], missing, axis=0), window], axis=0)
        return torch.from_numpy(np.asarray(window, dtype=np.float32))


def load_file_list(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File list does not exist: {file_path}")
    frame = pd.read_csv(file_path)
    if "file_name" not in frame.columns:
        raise ValueError("File list must contain a 'file_name' column")
    names = [str(value) for value in frame["file_name"].dropna().tolist()]
    if not names:
        raise ValueError("File list is empty")
    if len(names) != len(set(names)):
        raise ValueError("File list contains duplicate filenames")
    return names
