# training/self_play.py

import random
import torch
from typing import List
from game_engine.board import BoardState
from game_engine.rules import get_legal_moves, apply_move, is_terminal
from game_engine.scoring import compute_final_scores
from rl_agents.mcts import MCTSNode, mcts_search
from rl_agents.policy_wrapper import encode_state, decode_policy

def run_self_play_game(model, n_mcts_sim=50, device='cpu'):
    """
    Runs a single 4-player self-play game using MCTS for each player's turn.
    Returns a list of (state_encoding, policy_target, value_target) for training.

    We'll store "policy_target" as a vector [22], which is the MCTS visit distribution
    over (21 piece-ids + pass). We'll store "value_target" as the final outcome 
    from the perspective of the current_player.

    We do naive multi-player: each state only gets labeled from the vantage of the 
    color to move at that step.
    """

    # Initialize
    st = BoardState()
    trajectory = []  # list of (board_t, piece_t, pi, current_player)

    # We'll keep a separate MCTS root node that updates each turn
    root = MCTSNode(st)
    done = False

    while not is_terminal(st):
        # run MCTS from root
        mcts_search(root, model, n_simulations=n_mcts_sim, c_puct=1.0, device=device)

        # gather visit counts for each child => policy distribution
        # We sum visits for each piece_id (and pass) across placements
        visits = {}
        total_visits = 0
        for action_key, child in root.children.items():
            # action_key could be (pid, idx) or ('pass', 0)
            vcount = child.visit_count
            total_visits += vcount
            if action_key[0] == 'pass':
                # accumulate visits to 'pass'
                visits['pass'] = visits.get('pass', 0) + vcount
            else:
                pid = action_key[0]
                visits[pid] = visits.get(pid, 0) + vcount

        # Create a 22-dim distribution (21 piece IDs + pass)
        pi = [0.0]*22
        for pid in range(21):
            if pid in visits:
                pi[pid] = visits[pid] / total_visits
        pass_idx = 21
        if 'pass' in visits:
            pi[pass_idx] = visits['pass'] / total_visits

        # record the training example
        board_t, piece_t = encode_state(st)
        current_player = st.current_player
        trajectory.append((board_t, piece_t, pi, current_player))

        # pick an action from the visits distribution
        # (some exploration approach: random choice by pi, or argmax)
        action_index = random.choices(range(22), weights=pi, k=1)[0]
        if action_index == 21:
            # pass
            mv = None
        else:
            # pick a legal placement from the expansions
            # we must sample among all child nodes with that piece_id
            # gather children
            piece_id = action_index
            valid_children = []
            for (ak, childnode) in root.children.items():
                if ak[0] == 'pass':
                    continue
                if ak[0] == piece_id:
                    valid_children.append((ak, childnode))
            if len(valid_children) == 0:
                # forced pass if no actual child
                mv = None
            else:
                # pick uniformly among them or pick the most visited
                # let's pick the child with the highest visit_count
                best_child = max(valid_children, key=lambda x: x[1].visit_count)
                ak_best = best_child[0]
                # We'll reconstruct the actual move by retrieving 
                # from apply_move in expand_node. 
                # But we didn't store that; let's do a hack: we can re-get legal moves, find the same index, etc.

                # Actually, simpler approach: we can store the state from best_child as next state.
                # We'll proceed that way.
                pass
            # We already have best_child. So let's define next_state
            next_node = best_child[1]
            st = next_node.state
            root = next_node
            root.parent = None
            continue

        # If we get here, that means the chosen action is pass or forced pass
        st = apply_move(st, mv)
        # find the child node for pass
        pass_key = ('pass', 0)
        if pass_key in root.children:
            root = root.children[pass_key]
            root.parent = None
        else:
            # no pass child => we create new node
            root = MCTSNode(st)

    # terminal
    final_scores = compute_final_scores(st)
    # e.g. final_scores[color] is negative leftover squares
    # We'll label each (turn) with that player's final score, or a scaled version
    # For example, we could just store final_scores[color] as is:
    samples = []
    for (board_t, piece_t, pi, pl_color) in trajectory:
        value_target = final_scores[pl_color]  # e.g. - leftover_squares
        # Could scale it to [-1..1], up to you
        # Let's keep it as is for now
        samples.append((board_t, piece_t, pi, float(value_target)))

    return samples

def generate_self_play_data(model, n_games, n_mcts_sim=50, device='cpu'):
    """
    Runs multiple self-play games, returns a combined list of training samples.
    Each sample: (board_t, piece_t, pi, value_target)
    """
    all_data = []
    for g in range(n_games):
        game_samples = run_self_play_game(model, n_mcts_sim=n_mcts_sim, device=device)
        all_data.extend(game_samples)
    return all_data
