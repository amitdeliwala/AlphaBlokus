# rl_agents/mcts.py

import math
import random
from typing import Dict, Tuple
import torch
import numpy as np

from game_engine.board import BoardState
from game_engine.rules import get_legal_moves, apply_move, is_terminal
from game_engine.scoring import compute_final_scores
from rl_agents.policy_wrapper import encode_state, decode_policy

class MCTSNode:
    """
    A node in the MCTS tree.
    Stores:
    - parent
    - children: dict { (piece_id, move_placement_idx): MCTSNode } or "pass"
    - prior_prob for each child
    - visit_count, total_value
    - if expanded
    """
    def __init__(self, state: BoardState, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # map from action_key -> child node
        self.visit_count = 0
        self.total_value = 0.0
        self.priors = {}  # map from action_key -> prior prob
        self.expanded = False

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

def ucb_score(parent, child, c_puct=1.0):
    """
    UCB for MCTS, similar to AlphaGo's formula:
    Q + c_puct * P * sqrt(parent_visits)/(1+child_visits)
    """
    if child.visit_count == 0:
        return float('inf')
    return child.mean_value + c_puct * child.priors.get(child.last_action_key, 0) * math.sqrt(parent.visit_count) / (1 + child.visit_count)

def mcts_search(root: MCTSNode, model, n_simulations=50, c_puct=1.0, device='cpu'):
    """
    Perform MCTS from the root node for n_simulations. Each simulation:
    1. Select
    2. Expand
    3. Evaluate
    4. Backup
    """
    for _ in range(n_simulations):
        node = root
        # 1) Select
        while node.expanded and len(node.children) > 0:
            # pick child with max UCB
            best_action, best_child = None, None
            best_score = -999999.0
            for action_key, child in node.children.items():
                score = ucb_score(node, child, c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action_key
                    best_child = child
            node = best_child
        # 2) Expand if not terminal
        if not node.expanded and not is_terminal(node.state):
            expand_node(node, model, device)
        # 3) Evaluate
        value = evaluate_node(node, model, device)
        # 4) Backup
        backup_value(node, value)

def expand_node(node: MCTSNode, model, device='cpu'):
    """
    Expand the node by:
    1) Use the network policy to get piece distribution
    2) For each piece (or pass), enumerate next states (especially all placements)
    3) Create child nodes
    """
    state = node.state
    if is_terminal(state):
        node.expanded = True
        return

    # get policy from network
    board_t, piece_t = encode_state(state)
    board_t = board_t.to(device)
    piece_t = piece_t.to(device)
    with torch.no_grad():
        policy_logits, _ = model(board_t, piece_t)
    policy_logits = policy_logits[0]  # shape [22]
    
    # legal moves
    legal_moves = get_legal_moves(state)
    # piece IDs that are still available
    legal_piece_ids = state.pieces_remaining[state.current_player]

    # decode policy into piece-level distribution
    piece_policy = decode_policy(policy_logits, legal_piece_ids, temperature=1.0)  
    # piece_policy is {pid: prob, 'pass': prob}

    # For each piece, get all placements from the rule-based "legal_moves" that match that piece.
    # We'll store them as distinct children with some ID: (piece_id, index_in_that_piece_moves).
    moves_by_piece = {}
    for mv in legal_moves:
        (pid, row_off, col_off, shape_coords) = mv
        moves_by_piece.setdefault(pid, []).append(mv)

    # We also check if pass is an option. If there are no legal moves, pass might be the only move.
    # But pass might appear in the policy distribution.
    pass_prob = piece_policy.get('pass', 0.0)

    # For each piece
    for pid in legal_piece_ids:
        pprob = piece_policy.get(pid, 0.0)
        if pid not in moves_by_piece:
            # If the net suggests this piece, but there's no legal orientation/placement -> ignore
            continue
        # We'll distribute that piece's prior prob among all the actual placements.
        placements = moves_by_piece[pid]
        if len(placements) == 0:
            # no actual placements
            continue
        split_prob = pprob / len(placements)
        for idx, mv in enumerate(placements):
            child_state = apply_move(state, mv)
            child_node = MCTSNode(child_state, parent=node)
            # We'll store in child_node a reference to its action_key for UCB
            action_key = (pid, idx)
            child_node.last_action_key = action_key
            node.children[action_key] = child_node
            node.priors[action_key] = split_prob

    # If pass is a real option (network or forced)
    if pass_prob > 0.0:
        # create pass child
        mv_pass = None
        pass_state = apply_move(state, mv_pass)
        pass_node = MCTSNode(pass_state, parent=node)
        pass_node.last_action_key = ('pass', 0)
        node.children[('pass', 0)] = pass_node
        node.priors[('pass', 0)] = pass_prob

    # Mark expanded
    node.expanded = True

def evaluate_node(node: MCTSNode, model, device='cpu'):
    """
    Evaluate the node's state:
    - if terminal, compute final score outcome from the perspective of node.state.current_player
    - otherwise, use the value network
    Returns a float in [-1,1] or so.
    """
    st = node.state
    if is_terminal(st):
        # compute final score
        final_scores = stt_final_scores(st)
        # vantage point is st.current_player at the moment the game ended, though 
        # in a multi-player game, you might define a different scheme. 
        # We'll do a naive approach: the "value" is final_scores[current_player] 
        # relative to others, or just normalized. Up to you.
        # For simplicity, let's do final_scores[current_player] / 100 
        # (arbitrary scaling).
        val = final_scores[st.current_player] / 100.0
        return val
    else:
        # Use the network value
        board_t, piece_t = encode_state(st)
        board_t = board_t.to(device)
        piece_t = piece_t.to(device)
        with torch.no_grad():
            _, value = model(board_t, piece_t)
        return value.item()

def backup_value(node: MCTSNode, value: float):
    """
    Propagate the value up the tree. 
    In a multi-player game, you'd rotate perspective or handle each parent's current_player differently.
    For simplicity, we'll just do a direct pass as if it's 2-player.
    A more thorough approach might invert 'value' if we pass to a different color, etc.
    """
    # This is a big question in multi-player MCTS: how to back up a single scalar.
    # We'll do a naive approach: each node uses value from the perspective of that node's current_player.
    # As we go up, the parent's current_player might differ. A typical approach is zero-sum for 2-player,
    # but for 4-player we might do something else. 
    # Here, we won't invert or rotate. We'll just store the same value for each ancestor. 
    # This isn't strictly correct for multi-player, but let's keep it simple.

    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.total_value += value
        cur = cur.parent

def stt_final_scores(st: BoardState):
    """
    Helper to call compute_final_scores properly. Returns list of length 4.
    """
    return compute_final_scores(st)
