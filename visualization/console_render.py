"""
console_render.py

Renders the board in ASCII so we can observe the game state in the console.
"""

from game_engine.board import BOARD_SIZE

COLOR_CHARS = ['R','B','Y','G']  # Red, Blue, Yellow, Green

def print_board(board_grid):
    """
    Print the 20x20 board. Each cell is either '.' if empty or R/B/Y/G if occupied.
    board_grid[r][c] is either None or 0..3
    """
    print("   " + "".join([f"{c%10}" for c in range(BOARD_SIZE)]))
    for r in range(BOARD_SIZE):
        row_str = []
        for c in range(BOARD_SIZE):
            occupant = board_grid[r][c]
            if occupant is None:
                row_str.append('.')
            else:
                row_str.append(COLOR_CHARS[occupant])
        # Print row index + row
        print(f"{r%10:2d} " + "".join(row_str))
    print()
