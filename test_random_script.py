"""
test_random_game.py

Demonstrates a complete random-agent 4-player Blokus game,
using the console visualization to show progress.

To run:
  python test_random_game.py
"""

import random
from game_engine import BoardState, get_legal_moves, apply_move, is_terminal, compute_final_scores
from visualization import print_board
from visualization.pygame_viewer import replay_game

def play_random_blokus_game():
    state = BoardState()
    states_timeline = []

    turn_number = 0

    while not is_terminal(state):
        turn_number += 1
        current_player = state.current_player
        legal_moves = get_legal_moves(state)

        if len(legal_moves) == 0:
            # No moves -> pass
            next_state = apply_move(state, None)
            print(f"Turn {turn_number}: Player {current_player} passes.")
        else:
            chosen_move = random.choice(legal_moves)
            next_state = apply_move(state, chosen_move)
            piece_id, row_off, col_off, shape_coords = chosen_move
            print(f"Turn {turn_number}: Player {current_player} places piece {piece_id} at ({row_off},{col_off}).")
        states_timeline.append(state.clone())
        state = next_state
        if turn_number % 5 == 0:  # Print board every few turns, or every turn if you like
            print_board(state.board_grid)

    # Game over
    print("Game finished!")
    final_scores = compute_final_scores(state)
    for color, score in enumerate(final_scores):
        print(f"Player {color} score: {score}")


    # After the game finishes:
    replay_game(states_timeline)

if __name__ == "__main__":
    play_random_blokus_game()
