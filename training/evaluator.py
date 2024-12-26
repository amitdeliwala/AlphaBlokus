# training/evaluator.py

import logging
import random
from game_engine.board import BoardState
from game_engine.rules import get_legal_moves, apply_move, is_terminal
from game_engine.scoring import compute_final_scores
from rl_agents.mcts import MCTSNode, mcts_search

def evaluate_vs_random(model, n_games=10, device='cpu'):
    """
    Plays n_games where the model always moves for the current_player
    but with some probability the move is random, or you can do half model vs half random, etc.
    This is a placeholder. A more robust approach would have multiple agents.

    Returns average final score for model as a single number (?), or distribution of outcomes.
    """

    logger = logging.getLogger("AlphaBlokusEval")
    logger.setLevel(logging.INFO)

    total_score_sum = 0.0
    for g in range(n_games):
        st = BoardState()
        while not is_terminal(st):
            cur_player = st.current_player
            # model picks a move with MCTS
            root = MCTSNode(st)
            mcts_search(root, model, n_simulations=30, device=device)
            # pick best move from child visits
            best_child = None
            best_visits = -1
            chosen_action = None
            for ak, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_child = child
                    chosen_action = ak
            if best_child is None:
                # pass
                st = apply_move(st, None)
            else:
                st = best_child.state

        final_scores = compute_final_scores(st)
        # We'll sum up the model's color score. But "the model" is playing all 4 colors in a single-net approach.
        # We'll just pick color 0 for measurement, or sum them all.
        # This is ambiguous in a single-net scenario. 
        # Let's just do color 0 as "our" color:
        total_score_sum += final_scores[0]

    avg_score = total_score_sum / n_games
    logger.info(f"Avg final score for color0 across {n_games} games = {avg_score}")
    return avg_score
