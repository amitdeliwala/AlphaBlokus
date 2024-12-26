"""
pygame_viewer.py

A PyGame-based replay viewer for Blokus, with a side panel showing:
- Current partial scores (based on leftover squares).
- How many pieces each color has left.
- Which specific piece IDs remain.

Usage:
1. Simulate a Blokus game, collecting states at each turn in a list.
2. Pass them to replay_game(states).
3. Use the left/right arrow keys to navigate turns, ESC or 'q' to quit.
"""

import pygame
import sys
from typing import List
from game_engine.board import BoardState, BOARD_SIZE
from game_engine.scoring import compute_final_scores
from game_engine.pieces import ALL_SHAPES

# Define a fixed size for each board cell (in pixels)
CELL_SIZE = 20

# Side panel width (in pixels)
SIDE_PANEL_WIDTH = 300

# Color mapping for each occupant on the board (0=Red, 1=Blue, 2=Yellow, 3=Green, None=White)
COLOR_FOR_OCCUPANT = {
    0: (255, 0, 0),       # Red
    1: (0, 0, 255),       # Blue
    2: (255, 255, 0),     # Yellow
    3: (0, 200, 0),       # Green
    None: (255, 255, 255) # Empty
}

# Basic background color for the entire window
BG_COLOR = (40, 40, 40)

# A convenient name for each color index
COLOR_NAMES = ["Red", "Blue", "Yellow", "Green"]


def draw_board(screen: pygame.Surface,
               board: BoardState,
               font: pygame.font.Font,
               turn_index: int,
               total_turns: int):
    """
    Draws the main 20x20 grid portion of the board onto the screen.
    Also draws a small label with the turn index on the top-left corner.
    """
    # Clear the background first
    screen.fill(BG_COLOR)

    # --- Draw the board grid on the left side ---
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            occupant = board.board_grid[r][c]
            color = COLOR_FOR_OCCUPANT[occupant]
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            # Grid line (optional)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    # Turn label on top-left
    label_surface = font.render(f"Turn {turn_index+1}/{total_turns}", True, (255, 255, 255))
    screen.blit(label_surface, (10, 10))


def draw_side_panel(screen: pygame.Surface,
                    board: BoardState,
                    font: pygame.font.Font,
                    turn_index: int,
                    total_turns: int):
    """
    Draws a side panel on the right, showing partial scores and
    remaining pieces for each color.
    """
    # We'll place the side panel immediately to the right of the board
    side_panel_x = BOARD_SIZE * CELL_SIZE

    # Draw a background rectangle for the side panel
    side_panel_rect = pygame.Rect(side_panel_x, 0, SIDE_PANEL_WIDTH, BOARD_SIZE * CELL_SIZE)
    pygame.draw.rect(screen, (50, 50, 50), side_panel_rect)

    # Compute partial scores by calling compute_final_scores
    partial_scores = compute_final_scores(board)

    # Start rendering text below some margin
    margin_x = side_panel_x + 10
    margin_y = 10

    # Title
    title_surf = font.render("Blokus Info", True, (255, 255, 255))
    screen.blit(title_surf, (margin_x, margin_y))
    margin_y += 40

    # For each color, show its name, partial score, pieces left, and the piece IDs
    for color in range(4):
        color_text = COLOR_NAMES[color]
        # partial score
        color_score = partial_scores[color]
        # how many pieces left
        pieces_left = board.pieces_remaining[color]

        # First line: e.g. "Red: score X"
        color_info_surf = font.render(
            f"{color_text} (C={color}): Score {color_score}", True, (255, 255, 255)
        )
        screen.blit(color_info_surf, (margin_x, margin_y))
        margin_y += 25

        # Second line: "# pieces left"
        pieces_count = len(pieces_left)
        pieces_left_surf = font.render(
            f"Pieces left: {pieces_count}",
            True,
            (200, 200, 200)
        )
        screen.blit(pieces_left_surf, (margin_x, margin_y))
        margin_y += 25

        # (Optional) Show the piece IDs themselves
        # We'll break them up if there's a bunch. Let's do a single line though:
        if pieces_count > 0:
            piece_list_str = ", ".join(str(pid) for pid in sorted(pieces_left))
            piece_ids_surf = font.render(
                f"IDs: {piece_list_str}",
                True,
                (150, 150, 150)
            )
            screen.blit(piece_ids_surf, (margin_x, margin_y))
            margin_y += 25

        margin_y += 15  # blank space

    # Show instructions
    instr1_surf = font.render("Left/Right arrow = Prev/Next turn", True, (255, 255, 255))
    screen.blit(instr1_surf, (margin_x, margin_y))
    margin_y += 25
    instr2_surf = font.render("Esc / Q = Quit", True, (255, 255, 255))
    screen.blit(instr2_surf, (margin_x, margin_y))
    margin_y += 25


def replay_game(states: List[BoardState]):
    """
    Opens a PyGame window and allows the user to step through the provided
    sequence of BoardState objects with arrow keys.
    """
    pygame.init()
    pygame.display.set_caption("Blokus Replay (with Side Panel)")

    # Determine the window size based on board dimensions plus side panel
    window_width = BOARD_SIZE * CELL_SIZE + SIDE_PANEL_WIDTH
    window_height = BOARD_SIZE * CELL_SIZE

    screen = pygame.display.set_mode((window_width, window_height))

    # A font for rendering text
    font = pygame.font.SysFont(None, 24)

    # We'll keep an integer index for which state we're currently displaying
    current_index = 0
    total_states = len(states)

    clock = pygame.time.Clock()

    while True:
        clock.tick(30)  # limit to 30 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_LEFT:
                    current_index = max(0, current_index - 1)
                elif event.key == pygame.K_RIGHT:
                    current_index = min(total_states - 1, current_index + 1)

        # Draw current state
        current_state = states[current_index]
        draw_board(screen, current_state, font, current_index, total_states)
        draw_side_panel(screen, current_state, font, current_index, total_states)

        pygame.display.flip()


def test_pygame_viewer():
    """
    Simple demonstration that fakes a small list of states and replays them.
    Replace with actual data from your game simulation.
    """
    from ..game_engine.board import BoardState
    import random
    random.seed(42)

    # Generate some dummy states
    states = []
    for i in range(5):
        bs = BoardState()
        # Fill random spots for demonstration
        for _ in range((i+1) * 5):
            rr = random.randint(0, BOARD_SIZE-1)
            cc = random.randint(0, BOARD_SIZE-1)
            color = random.randint(0,3)
            bs.board_grid[rr][cc] = color
        # Randomly remove some pieces
        for color in range(4):
            remove_count = random.randint(0, 3)
            for _ in range(remove_count):
                if len(bs.pieces_remaining[color]) > 0:
                    any_piece = random.choice(list(bs.pieces_remaining[color]))
                    bs.pieces_remaining[color].remove(any_piece)
        states.append(bs)

    replay_game(states)

if __name__ == "__main__":
    test_pygame_viewer()
