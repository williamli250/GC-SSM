"""Command-line interface for GC-SSM experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

from gc_ssm.config import load_config


DATASET_DIR = Path("data") / "TSB-AD-M"
FILE_LIST = Path("data") / "File_List" / "TSB-AD-M-Eva.csv"
OUTPUT_DIR = Path("output")


def _batch_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_argument_group("model overrides")
    model.add_argument("--d-model", type=int)
    model.add_argument("--d-state", type=int)
    model.add_argument("--d-conv", type=int)
    model.add_argument("--expand", type=int)
    model.add_argument("--num-blocks", type=int)
    model.add_argument("--dropout", type=float)
    model.add_argument("--node-emb-dim", type=int)
    model.add_argument("--random-feature-dim", type=int)
    model.add_argument("--random-feature-seed", type=int)
    model.add_argument("--forecast-steps", type=int, nargs="+")

    training = parser.add_argument_group("training overrides")
    training.add_argument("--lambda-forecast", type=float)
    training.add_argument("--lr", type=float)
    training.add_argument("--weight-decay", type=float)
    training.add_argument("--epochs", type=int)
    training.add_argument("--patience", type=int)
    training.add_argument("--val-ratio", type=float)
    training.add_argument("--batch-size", type=_batch_size)
    training.add_argument("--seed", type=int)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"))
    training.add_argument("--model-window", type=int)


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "d_model",
        "d_state",
        "d_conv",
        "expand",
        "num_blocks",
        "dropout",
        "node_emb_dim",
        "random_feature_dim",
        "random_feature_seed",
        "forecast_steps",
        "lambda_forecast",
        "lr",
        "weight_decay",
        "epochs",
        "patience",
        "val_ratio",
        "batch_size",
        "seed",
        "device",
        "model_window",
    )
    return {name: getattr(args, name, None) for name in names}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gc_ssm",
        description="Run GC-SSM on all 180 TSB-AD-M evaluation files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_override_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from gc_ssm.experiment import benchmark_experiment

        config = load_config(_config_overrides(args))
        summary = benchmark_experiment(
            DATASET_DIR, FILE_LIST, config, OUTPUT_DIR
        )
        print(
            f"Completed {summary['n_completed']}/{summary['n_expected']} files "
            f"with {summary['n_errors']} errors"
        )
        print(f"Summary: {OUTPUT_DIR / 'summary.json'}")
        return 1 if summary["n_errors"] else 0
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
