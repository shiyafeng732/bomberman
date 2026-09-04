"""
Model 2: Deep Q-Network (DQN) in PyTorch.

Instead of one linear weight vector per action we approximate

    Q(s, .) = MLP(phi(s))

with a small fully connected network (26 -> 128 -> 128 -> 6). The input is
exactly the same feature vector as for Model 1, so any difference in playing
strength comes from the model, not from the features.

The network is small on purpose: the tournament runs on the CPU.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from .features import ACTIONS, FEATURE_DIM, state_to_features

MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-model.pt")

HIDDEN_SIZE = 128

# Probabilities used while exploring: walking is more useful than waiting/bombing.
EXPLORATION_PROBS = np.array([0.20, 0.20, 0.20, 0.20, 0.10, 0.10])


class DQN(nn.Module):
    """Small multilayer perceptron mapping features to one Q value per action."""

    def __init__(self, input_dim=FEATURE_DIM, hidden=HIDDEN_SIZE, n_actions=len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def setup(self):
    """Called once when the agent is loaded. Prepares self.model."""
    self.device = torch.device("cpu")          # the tournament runs on the CPU
    self.model = DQN().to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading DQN weights from file.")
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
    else:
        self.logger.info("Setting up a fresh DQN.")

    self.model.eval()


def q_values(self, features):
    """Q value of every action in the given state, as a numpy array."""
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        return self.model(tensor).numpy()


def act(self, game_state: dict) -> str:
    """Choose an action: epsilon-greedy while training, greedy otherwise."""
    features = state_to_features(game_state)

    epsilon = getattr(self, "epsilon", 0.0) if self.train else 0.0
    if np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS, p=EXPLORATION_PROBS)
        self.logger.debug(f"Exploring: {action}")
        return action

    q = q_values(self, features)
    # Break ties randomly, otherwise the agent always prefers the first action.
    best = np.flatnonzero(q == q.max())
    action = ACTIONS[np.random.choice(best)]
    self.logger.debug(f"Greedy action {action} with Q = {q.round(2)}")
    return action
