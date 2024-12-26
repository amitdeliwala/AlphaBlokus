# rl_agents/policy_wrapper.py

import torch
import numpy as np
from game_engine.board import BoardState, BOARD_SIZE
from game_engine.rules import get_legal_moves

def encode_state(board_state: BoardState) -> (torch.Tensor, torch.Tensor):
    """
    Convert board_state into (board_channels, piece_vector).
    board_channels shape: [5, 20, 20]
    piece_vector shape: [21]

    - Channels 0..3: Occupancy by color c
    - Channel 4: 1 if cell is occupied by board_state.current_player, else 0
    - piece_vector[i] = 1 if piece i is STILL available for current_player, else 0
    """
    current_player = board_state.current_player

    board_channels = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Fill occupancy channels
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            occupant = board_state.board_grid[r][c]
            if occupant is not None:
                board_channels[occupant, r, c] = 1.0
                # occupant in [0..3]

    # Channel 4: highlight squares of the current_player
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            occupant = board_state.board_grid[r][c]
            if occupant == current_player:
                board_channels[4, r, c] = 1.0

    # piece_vector
    piece_vec = np.zeros((21,), dtype=np.float32)
    for p_id in board_state.pieces_remaining[current_player]:
        piece_vec[p_id] = 1.0

    # Convert to torch tensors
    board_t = torch.from_numpy(board_channels).unsqueeze(0)  # shape [1, 5, 20, 20]
    piece_t = torch.from_numpy(piece_vec).unsqueeze(0)       # shape [1, 21]

    return board_t, piece_t

def decode_policy(policy_logits, legal_piece_ids, temperature=1.0):
    """
    Convert model's [22]-dim policy logits into a distribution over (21 pieces + pass).
    We then filter out any piece_ids not in legal_piece_ids. If none are legal, pass is forced.

    We use a softmax with 'temperature' for exploration. 
    If temperature=0.0, it's effectively an argmax.

    Returns a dictionary {action_id: probability}, where action_id in [0..20 or 'pass'].
    """

    # policy_logits: shape [22], containing 21 pieces + 1 pass
    # legal_piece_ids: set of piece IDs that remain for the current player
    # We'll do a numpy softmax

    logits_np = policy_logits.detach().cpu().numpy()  # shape [22]
    # Filter out piece IDs that are not in legal_piece_ids
    # We'll also handle the pass action (index = 21)
    # We'll create a mask
    mask = np.zeros_like(logits_np, dtype=bool)
    for pid in legal_piece_ids:
        mask[pid] = True
    # Mark the pass action as always legal
    pass_index = 21
    mask[pass_index] = True

    # If no legal pieces (empty set?), then pass is forced
    if len(legal_piece_ids) == 0:
        # produce a dictionary that pass=1.0
        return { 'pass': 1.0 }

    # masked logits
    masked_logits = np.full_like(logits_np, -1e9)  # large negative
    masked_logits[mask] = logits_np[mask]

    # Apply temperature
    if temperature > 1e-8:
        scaled_logits = masked_logits / temperature
        exps = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exps / np.sum(exps)
    else:
        # Argmax
        probs = np.zeros_like(masked_logits)
        probs[np.argmax(masked_logits)] = 1.0

    out_dict = {}
    for i in range(21):
        if i in legal_piece_ids:
            out_dict[i] = probs[i]
    out_dict['pass'] = probs[pass_index]

    return out_dict
