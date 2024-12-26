# rl_agents/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaBlokusNet(nn.Module):
    """
    Combined policy + value network for Blokus.
    Inputs:
      - board_channels: shape [batch, 5, 20, 20]
      - piece_vector: shape [batch, 21] (which pieces remain for the current player)

    Outputs:
      - policy_logits: shape [batch, 22]  (21 pieces + 1 pass action)
      - value: shape [batch] (scalar)
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers for the 5-channel board
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # Possibly add more layers or residual blocks in a real project

        # We'll flatten the conv output and then combine with the piece_vector
        # The board is 20x20, after 3 conv layers, shape is still [64, 20, 20]
        # We'll do a global average pooling or flatten
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Now we combine the pooled conv features (64 dims) with the 21-dim piece vector
        # So total = 64 + 21 = 85
        self.fc_combine = nn.Linear(32 + 21, 128)

        # Policy head
        self.policy_head = nn.Linear(128, 22)  # 21 pieces + 1 pass
    
        # Value head
        self.value_head1 = nn.Linear(128, 32)
        self.value_head2 = nn.Linear(32, 1)

    def forward(self, board_channels, piece_vector):
        """
        board_channels: [batch, 5, 20, 20] float or bool
        piece_vector:   [batch, 21] float or bool
        """
        x = F.relu(self.conv1(board_channels))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Global average pool
        # shape -> [batch, 64, 1, 1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # [batch, 64]

        # Concatenate piece_vector
        x = torch.cat([x, piece_vector], dim=1)  # -> [batch, 64 + 21]

        x = F.relu(self.fc_combine(x))  # -> [batch, 128]

        # Policy
        policy_logits = self.policy_head(x)  # [batch, 22]

        # Value
        vh = F.relu(self.value_head1(x))      # [batch, 64]
        value = self.value_head2(vh)          # [batch, 1]
        value = torch.tanh(value)             # range -1..1, or do nothing if you prefer

        return policy_logits, value.squeeze(-1)
