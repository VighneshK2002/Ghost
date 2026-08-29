"""Vectorized non-vision delayed-cue T-maze.

The implementation is re-exported from the behavior-preserving research
engine. Keeping one definition avoids subtle environment drift between the
historical Marimo interface and the command-line experiment.
"""

from ghost.ghost_terminal_core import ACTIONS, BatchedTMaze

__all__ = ["ACTIONS", "BatchedTMaze"]
