"""
rules.py

Contains functions to:
- Generate all legal moves for a given BoardState and player
- Apply a move (place a piece) onto the board
- Check terminal condition (4 consecutive passes or all pieces placed)
"""

from typing import List, Tuple, Optional, Set
from .board import BoardState, BOARD_SIZE
from .pieces import ALL_SHAPES, get_all_orientations, normalize_shape

Move = Tuple[int, int, int, Set[Tuple[int, int]]] 
# We'll define a Move as (piece_id, row_offset, col_offset, shape_coords)
# where shape_coords is the chosen orientation's squares,
# and (row_offset, col_offset) is how we shift them onto the board.

CORNER_POSITIONS = [(0,0), (0,BOARD_SIZE-1), (BOARD_SIZE-1,0), (BOARD_SIZE-1, BOARD_SIZE-1)]

def check_legal_move(state: BoardState, move: Move) -> bool:
    """
    Check if placing the given piece in the given orientation/position is legal:
      1) piece is in player's remaining set
      2) if it's the player's first move, it covers that player's corner
      3) squares are within board and not occupied
      4) it does not share an edge with same-color squares
      5) it must share at least one corner with same-color squares (except if it's the first move, then must cover corner).
    """
    piece_id, row_off, col_off, shape_coords = move
    color = state.current_player

    # 1) piece must be available
    if piece_id not in state.pieces_remaining[color]:
        return False

    # We'll gather the absolute board squares
    placed_squares = []
    for (r, c) in shape_coords:
        rr = r + row_off
        cc = c + col_off
        # 3) within board check
        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
            return False
        # not already occupied
        if state.board_grid[rr][cc] is not None:
            return False
        placed_squares.append((rr, cc))

    # If it's the first move for this color, we must cover the color's corner
    # Let's define corner index = color for simplicity:
    if state.moves_made[color] == 0:
        corner_r, corner_c = CORNER_POSITIONS[color]
        # Check if (corner_r, corner_c) is in placed_squares
        if (corner_r, corner_c) not in placed_squares:
            return False
        # If the first move passes these checks, it's legal.
        return True

    # After first move, we must:
    # (a) Not share an edge with same color squares
    # (b) Must share at least a corner with same color squares

    # We'll track adjacency
    corner_contact = False

    # Directions for edge adjacency
    edge_neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    # Directions for corner adjacency
    corner_neighbors = [(1,1),(1,-1),(-1,1),(-1,-1)]

    # We'll convert placed squares to a set for quick lookup
    placed_set = set(placed_squares)

    for (rr, cc) in placed_squares:
        # Check all neighbors
        for dr, dc in edge_neighbors:
            nr, nc = rr+dr, cc+dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if state.board_grid[nr][nc] == color:
                    # if same color is on an edge, illegal
                    return False

        for dr, dc in corner_neighbors:
            nr, nc = rr+dr, cc+dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if state.board_grid[nr][nc] == color:
                    corner_contact = True

    if not corner_contact:
        # Must share at least one corner with same color
        return False

    return True


def get_legal_moves(state: BoardState) -> List[Move]:
    """
    Enumerate all possible moves for state.current_player.
    If none are found, returning an empty list indicates a pass.
    """
    color = state.current_player
    legal_moves = []

    # The pieces that color still has
    available_pieces = state.pieces_remaining[color]

    for piece_id in available_pieces:
        shape = ALL_SHAPES[piece_id]
        # Get all orientations
        orientations = get_all_orientations(shape)

        for orientation in orientations:
            # orientation is a set of (r, c)
            # We'll attempt to place its top-left corner all over the board
            # but we have to consider all possible row_off/col_off shifts
            # that keep squares in [0..19].
            # Actually, we won't do a naive approach of 20x20 per orientation, 
            # because we have to still check legality anyway. That’s fine for now.

            # Just to be consistent, orientation is normalized so min row,col = 0,0
            # So the bounding box of orientation is something:
            max_r = max(r for r, c in orientation)
            max_c = max(c for r, c in orientation)

            # The shift that puts (0,0) at (row_off, col_off)
            # must keep (max_r, max_c) within board.
            for row_off in range(BOARD_SIZE - max_r):
                for col_off in range(BOARD_SIZE - max_c):
                    move_candidate = (piece_id, row_off, col_off, orientation)
                    if check_legal_move(state, move_candidate):
                        legal_moves.append(move_candidate)

    return legal_moves


def apply_move(state: BoardState, move: Optional[Move]) -> BoardState:
    """
    Returns a new BoardState after applying the given move.
    If move is None, it means pass.
    """
    new_state = state.clone()
    color = new_state.current_player

    if move is None:
        # pass
        new_state.consecutive_passes += 1
    else:
        # place piece
        piece_id, row_off, col_off, shape_coords = move
        new_state.pieces_remaining[color].remove(piece_id)

        for (r, c) in shape_coords:
            rr = r + row_off
            cc = c + col_off
            new_state.board_grid[rr][cc] = color

        new_state.moves_made[color] += 1
        new_state.consecutive_passes = 0

    # Switch to next player
    new_state.current_player = (color + 1) % 4

    return new_state


def is_terminal(state: BoardState) -> bool:
    """
    The game ends if:
    1) 4 consecutive passes have occurred, or
    2) All pieces of all players have been placed (extremely rare).
    """
    # Check consecutive passes
    if state.consecutive_passes >= 4:
        return True

    # Check if any player still has any piece
    # If all sets are empty, game ends
    for color in range(4):
        if len(state.pieces_remaining[color]) > 0:
            return False
    return True
