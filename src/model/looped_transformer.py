"""Team-facing import for the completed Looped/Recurrent Transformer."""

from ..original.model import RecurrentTransformer


LoopedTransformer = RecurrentTransformer

__all__ = ["LoopedTransformer", "RecurrentTransformer"]
