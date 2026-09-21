"""
Model 1: Q-learning with linear function approximation.

    Q(s, a) = w_a . phi(s)

phi(s) is the 26-dimensional feature vector from features.py and we keep one
weight vector per action, i.e. the whole model is a (6, 26) matrix. This is the
"lecture" model: temporal-difference reinforcement learning on top of a linear
regression model.

This file only contains the *playing* code. The learning happens in train.py.
"""

import os
import pickle

import numpy as np

from .features import ACTIONS, FEATURE_DIM, state_to_features

MODEL_FILE = os.path.join(os.path.dirname(__file__), "my-saved-model.pt")

# Probabilities used while exploring: walking is more useful than waiting/bombing.
EXPLORATION_PROBS = np.array([0.20, 0.20, 0.20, 0.20, 0.10, 0.10])


def setup(self):
    """Called once when the agent is loaded. Prepares self.model."""
    # We always continue from the stored weights when they exist. That is what
    # makes the curriculum (coins -> crates -> opponents) work: every training
    # stage starts where the previous one stopped. Delete the file to start over.
    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading Linear weights from file.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)
    elif self.train:
        self.logger.info("Setting up a fresh Linear model.")
    else:
        raise FileNotFoundError(f"{MODEL_FILE} is missing, refusing to play with random weights")



def q_values(self, features):
    """Q value of every action in the given state."""
    return self.model @ features


def act(self, game_state: dict) -> str:
    """Choose an action: epsilon-greedy while training, greedy otherwise."""
    features = state_to_features(game_state)

    epsilon = getattr(self, "epsilon", 0.0) if self.train else 0.0
    if np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS, p=EXPLORATION_PROBS)
        self.logger.debug(f"Exploring: {action}")
        return action

    q = q_values(self, features)
    allowed = np.ones(len(ACTIONS), dtype=bool)
    allowed[:4] = features[:4] > 0
    allowed[5] = features[21] > 0
    
    q = np.where(allowed, q, -np.inf)
    action = ACTIONS[np.random.choice(best)]
    self.logger.debug(f"Greedy action {action} with Q = {q.round(2)}")
    return action
