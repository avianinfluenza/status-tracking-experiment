"""Models and experiments that reproduce the team's original ball-swap plan.

This package is deliberately independent from :mod:`src.systematic`.  The
``systematic`` package implements the expanded research plan, while this one
keeps the original five-person, swap-length-generalization scope intact.
"""

from .model import (
    DirectTransformer,
    ExplicitCoTTransformer,
    OriginalModelConfig,
    RecurrentTransformer,
    build_model,
)

__all__ = [
    "DirectTransformer",
    "ExplicitCoTTransformer",
    "OriginalModelConfig",
    "RecurrentTransformer",
    "build_model",
]
