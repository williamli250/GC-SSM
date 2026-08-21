"""GC-SSM package."""

from typing import Any

__all__ = ["ExperimentConfig", "GCSSM", "ModelOutput"]


def __getattr__(name: str) -> Any:
    if name == "ExperimentConfig":
        from gc_ssm.config import ExperimentConfig

        return ExperimentConfig
    if name in {"GCSSM", "ModelOutput"}:
        from gc_ssm.models import GCSSM, ModelOutput

        return {"GCSSM": GCSSM, "ModelOutput": ModelOutput}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
