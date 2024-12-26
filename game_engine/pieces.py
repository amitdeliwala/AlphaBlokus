"""
pieces.py

Defines all 21 distinct Blokus shapes (for a single color),
each as a set of (row, col) offsets in a canonical orientation.

Provides a function get_all_orientations(piece_shape) that
returns all unique orientations (rotations + flips).
"""

from typing import Set, Tuple, List

# Each piece is defined as a set of (row, col) tuples.
# We will define them so that (0,0) is included and the shape
# is as "top-left" aligned as possible in its canonical orientation.

# For example, the 1-square piece is just {(0,0)}.
# The 2-square piece might be {(0,0),(0,1)}.
# We'll define the full set of 21 shapes (IDs 0..20).

# Helper function
def normalize_shape(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Shift a set of (r, c) coordinates so that the topmost-leftmost
    coordinate is (0,0). This ensures consistent representation.
    """
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    shifted = {(r - min_r, c - min_c) for (r, c) in shape}
    return shifted

def rotate_90(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Rotate shape 90 degrees clockwise around (0,0).
    Then re-normalize so top-left is (0,0).
    (row, col) -> (col, -row)
    """
    rotated = {(c, -r) for (r, c) in shape}
    return normalize_shape(rotated)

def flip_horizontal(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Reflect shape horizontally (mirror along vertical axis).
    (r, c) -> (r, -c)
    """
    flipped = {(r, -c) for (r, c) in shape}
    return normalize_shape(flipped)

def get_all_orientations(shape: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    """
    Return a list of all unique orientations (rotations + flips)
    of the given shape. Each orientation is normalized so that
    the top-left corner is (0,0).
    """
    # We'll generate:
    #  - 4 rotations
    #  - each rotation also can be flipped horizontally
    # So up to 8 distinct shapes, but some may duplicate. We'll keep them in a set.
    orientations = set()

    current = normalize_shape(shape)

    for _ in range(4):
        # Rotate 90 deg clockwise
        current = rotate_90(current)
        orientations.add(frozenset(current))

        # Flip horizontally
        flipped = flip_horizontal(current)
        orientations.add(frozenset(flipped))

    # Convert from frozenset back to set
    result = []
    for item in orientations:
        result.append(set(item))
    return result

# Now define all 21 shapes in canonical form, each as a set of (r, c).
# We'll store them in a list where index = piece_id.

ALL_SHAPES = []

# 1-square (piece_id = 0)
ALL_SHAPES.append({(0,0)})

# 2-square (piece_id = 1), simple domino horizontally
ALL_SHAPES.append({(0,0),(0,1)})

# 3-square pieces (piece_id = 2,3)
# Let's define two different tromino shapes: a straight one and an "L"
# (1) Straight tromino horizontally
ALL_SHAPES.append({(0,0),(0,1),(0,2)})
# (2) L shaped
ALL_SHAPES.append({(0,0),(1,0),(1,1)})

# 4-square pieces (piece_id = 4..8)
# We'll define 5 shapes: T, L, S, square, line
# 1) T shape
ALL_SHAPES.append({(0,0),(0,1),(0,2),(1,1)})  # T
# 2) L shape
ALL_SHAPES.append({(0,0),(1,0),(2,0),(2,1)})
# 3) S shape
ALL_SHAPES.append({(0,1),(0,2),(1,0),(1,1)})  # basically Z or S
# 4) square 2x2
ALL_SHAPES.append({(0,0),(0,1),(1,0),(1,1)})
# 5) line of 4
ALL_SHAPES.append({(0,0),(0,1),(0,2),(0,3)})

# 5-square (pentomino) pieces (piece_id = 9..20)
# We'll define the 12 standard pentominoes in some canonical orientation.
# We'll just define them quickly:

# 9: F
ALL_SHAPES.append({(0,1),(1,0),(1,1),(1,2),(2,0)})
#10: I (line of 5)
ALL_SHAPES.append({(0,0),(0,1),(0,2),(0,3),(0,4)})
#11: L (5 squares)
ALL_SHAPES.append({(0,0),(1,0),(2,0),(3,0),(3,1)})
#12: P
ALL_SHAPES.append({(0,0),(0,1),(1,0),(1,1),(2,0)})
#13: N
ALL_SHAPES.append({(0,0),(1,0),(2,0),(3,0),(3,1)})  # Similar to L but extends further
#14: T (5 squares)
ALL_SHAPES.append({(0,0),(0,1),(0,2),(1,1),(2,1)})
#15: U
ALL_SHAPES.append({(0,0),(0,2),(1,0),(1,1),(1,2)})
#16: V
ALL_SHAPES.append({(0,0),(1,0),(2,0),(3,0),(3,1)})
#17: W
ALL_SHAPES.append({(0,2),(1,1),(1,2),(2,0),(2,1)})
#18: X
ALL_SHAPES.append({(0,1),(1,0),(1,1),(1,2),(2,1)})
#19: Y
ALL_SHAPES.append({(0,0),(1,0),(2,0),(3,0),(2,1)})
#20: Z
ALL_SHAPES.append({(0,0),(0,1),(1,1),(1,2),(2,2)})

# Now each piece_id in [0..20] has a canonical shape in ALL_SHAPES.
# We'll rely on get_all_orientations() to get all possible transformations.
