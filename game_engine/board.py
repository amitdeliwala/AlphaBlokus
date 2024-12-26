"""
board.py

Defines the BoardState class for Blokus, including:
- A 20x20 board grid
- Which pieces each of the 4 players still hold
- Whose turn it is
"""

from typing import List, Optional
import copy

NUM_PLAYERS = 4
BOARD_SIZE = 20

class BoardState:
    """
    Stores the state of the Blokus board:
      - board_grid[r][c] = None if empty, or an integer (0..3) indicating which color occupies that cell.
      - pieces_remaining[color] = set of piece_ids (0..20) that that color has not placed yet
      - current_player = integer in [0..3]
      - moves_made[color] = how many moves the color has made (used to check if corner placement is still required)
    """

    def __init__(self):
        # 20x20 board, each cell is None if empty, else an integer 0..3
        self.board_grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        # For each color, we track which pieces remain. Each color has 21 unique piece_ids: 0..20
        self.pieces_remaining = [set(range(21)) for _ in range(NUM_PLAYERS)]

        # Start with player 0
        self.current_player = 0

        # Track how many turns each player has taken (to handle corner requirement on first move)
        self.moves_made = [0]*NUM_PLAYERS

        # Keep track of how many consecutive "no-move" passes happen
        # If it reaches 4 in a row, the game ends
        self.consecutive_passes = 0

    def clone(self):
        """
        Create a deep copy of the current board state.
        """
        new_state = BoardState()
        new_state.board_grid = copy.deepcopy(self.board_grid)
        new_state.pieces_remaining = [s.copy() for s in self.pieces_remaining]
        new_state.current_player = self.current_player
        new_state.moves_made = self.moves_made[:]
        new_state.consecutive_passes = self.consecutive_passes
        return new_state
