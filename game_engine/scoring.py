"""
scoring.py

Implements final scoring for Blokus.
Basic approach: Count squares left in each player's hand.
Optionally handle advanced scoring (+15 if placed all, +5 if last piece was monomino).
"""

from .board import BoardState, BOARD_SIZE

def compute_final_scores(state: BoardState) -> list:
    """
    Returns a list of 4 scores, one for each color, using advanced scoring.
    1 square left in hand = -1 point
    If all pieces placed, +15 points
    If the last piece placed was the single-square piece AND no pieces left, +5 more
    (We do not track which was the actual last piece placed for each color in this code, 
     so let's do a simpler approach for demonstration.)
    
    For demonstration, we'll do "basic scoring": each leftover square = -1 point,
    ignoring the bonus logic. Feel free to expand as desired.
    """
    scores = [0, 0, 0, 0]
    # We do not actually track how many squares each piece had in this minimal version,
    # so let's approximate: The piece_id correlates to the shape. We can count squares from the piece shape.

    # Mapping piece_id -> number_of_squares
    # This is rough: we can get it from the shape length.
    from .pieces import ALL_SHAPES

    for color in range(4):
        leftover_squares = 0
        for pid in state.pieces_remaining[color]:
            leftover_squares += len(ALL_SHAPES[pid])
        scores[color] = -leftover_squares  # each leftover square is -1
    return scores
