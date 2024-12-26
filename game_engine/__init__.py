"""
__init__.py

Empty initializer or re-exports.
"""

from .board import BoardState
from .pieces import ALL_SHAPES, get_all_orientations
from .rules import get_legal_moves, apply_move, is_terminal
from .scoring import compute_final_scores

__all__ = [
    "BoardState",
    "ALL_SHAPES",
    "get_all_orientations",
    "get_legal_moves",
    "apply_move",
    "is_terminal",
    "compute_final_scores"
]
