"""Shared model components for the ball-swap experiment.

Import Basic and Looped models from ``basic_transformer`` and
``looped_transformer`` respectively.  Keeping those imports out of this module
avoids a cycle while the implementations use the shared classifier.
"""

from .classifier import Classifier

__all__ = ["Classifier"]
