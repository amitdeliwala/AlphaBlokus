# training/replay_buffer.py

import random
import torch

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []

    def push(self, samples):
        """
        samples is a list of (board_t, piece_t, pi, value)
        - board_t: shape [1,5,20,20]
        - piece_t: shape [1,21]
        - pi: a list of length 22
        - value: float
        """
        for s in samples:
            self.buffer.append(s)
        while len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # remove oldest

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # We'll collate them
        board_batch = []
        piece_batch = []
        pi_batch = []
        value_batch = []
        for (b_t, p_t, pi, val) in batch:
            board_batch.append(b_t)
            piece_batch.append(p_t)
            pi_batch.append(pi)
            value_batch.append(val)

        # stack
        board_batch = torch.cat(board_batch, dim=0)  # shape [batch_size, 5, 20, 20]
        piece_batch = torch.cat(piece_batch, dim=0)  # shape [batch_size, 21]
        pi_batch = torch.tensor(pi_batch, dtype=torch.float32)  # [batch_size, 22]
        value_batch = torch.tensor(value_batch, dtype=torch.float32)  # [batch_size]

        return board_batch, piece_batch, pi_batch, value_batch

    def __len__(self):
        return len(self.buffer)
