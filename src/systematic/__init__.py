"""Controlled natural-language state-tracking research pipeline."""

from .data import (
    StateTrackingDataset,
    StateTrackingExample,
    StateTrackingGenerator,
    TokenVocabulary,
    collate_examples,
)
from .model import ModelConfig, StateTrackingTransformer, count_parameters
from .analysis import bootstrap_mean_ci, spearman_rho

__all__ = [
    "ModelConfig",
    "StateTrackingDataset",
    "StateTrackingExample",
    "StateTrackingGenerator",
    "StateTrackingTransformer",
    "TokenVocabulary",
    "collate_examples",
    "count_parameters",
    "bootstrap_mean_ci",
    "spearman_rho",
]
