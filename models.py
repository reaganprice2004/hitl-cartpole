"""
models.py - (Optional) separate model definitions.
Left minimal because agent.py currently defines a compact Q-network inline.
You can expand this file to hold larger architectures.
"""

import torch.nn as nn

class MLPQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)
