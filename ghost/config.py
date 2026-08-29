"""Experiment configuration.

The dataclass remains defined in the preserved research engine so saved
configuration dictionaries and training behavior retain one source of truth.
"""

from .ghost_terminal_core import Config

__all__ = ["Config"]
