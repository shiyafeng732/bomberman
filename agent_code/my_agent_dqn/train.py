"""
Training code for Model 2 (Deep Q-Network).

Differences to Model 1, and the reason why we expect it to be stronger:
  * experience replay      -- we learn from a random batch of past transitions
                              instead of only the last one, which removes the
                              correlation between consecutive samples
  * target network         -- the bootstrap value comes from a frozen copy of the
                              network, which stabilises the moving target
  * non-linear model       -- the MLP can represent interactions between features
                              (e.g. "coin is left" AND "left tile is dangerous")

Rewards and auxiliary events are identical to Model 1 on purpose, so the two
models can be compared fairly.
"""

import csv
import json
import os
import random
from collections import deque, namedtuple
from typing import List

import numpy as np
import torch
import torch.nn as nn

import events as e
from .callbacks import DQN, MODEL_FILE
from .features import ACTIONS, coin_distance, crate_distance, state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ---------------------------------------------------------------- hyperparameters
GAMMA = 0.95              # discount factor
LEARNING_RATE = 5e-4      # Adam step size
BATCH_SIZE = 64           # transitions per gradient step
BUFFER_SIZE = 50000       # size of the experience replay memory
LEARN_START = 1000        # collect this many transitions before the first update
TARGET_UPDATE = 500       # copy weights to the target network every ... steps

EPSILON_START = float(os.environ.get("EPSILON_START", 1.0))
EPSILON_END = 0.05
EPSILON_DECAY = 0.995     # multiplied with epsilon after every round

LOG_EVERY = 100
STATS_FILE = os.path.join(os.path.dirname(__file__), "training_stats.csv")

# ---------------------------------------------------------------- custom events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"
ESCAPED_DANGER = "ESCAPED_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
GOOD_BOMB = "GOOD_BOMB"
USELESS_BOMB = "USELESS_BOMB"
SUICIDAL_BOMB = "SUICIDAL_BOMB"

# ---------------------------------------------------------------- reward table
REWARDS = {
    e.COIN_COLLECTED: 5.0,
    e.KILLED_OPPONENT: 30.0,
    e.CRATE_DESTROYED: 1.0,
    e.COIN_FOUND: 0.5,
    e.SURVIVED_ROUND: 3.0,
    e.KILLED_SELF: -20.0,     # fires together with GOT_KILLED, so a suicide costs -50 in total
    e.GOT_KILLED: -30.0,
    e.INVALID_ACTION: -3.0,
    e.WAITED: -1.0,
    MOVED_TOWARDS_COIN: 1.0,
    MOVED_AWAY_FROM_COIN: -1.2,
    MOVED_TOWARDS_CRATE: 0.3,
    MOVED_AWAY_FROM_CRATE: -0.4,
    ESCAPED_DANGER: 3.0,
    STAYED_IN_DANGER: -1.0,
    MOVED_INTO_DANGER: -4.0,
    GOOD_BOMB: 10.0,
    USELESS_BOMB: -1.0,
    SUICIDAL_BOMB: -5.0,
}

# Controlled experiments override single reward values through the environment,
# so a sweep changes exactly one number and never edits this file:
#   REWARD_OVERRIDES='{"GOOD_BOMB": 3}' python main.py play ...
for _event, _value in json.loads(os.environ.get("REWARD_OVERRIDES", "{}")).items():
    if _event not in REWARDS:
        raise KeyError(f"REWARD_OVERRIDES: {_event!r} is not one of {sorted(REWARDS)}")
    REWARDS[_event] = float(_value)


def setup_training(self):
    """Called once after setup() when the agent runs in training mode."""
    self.model.train()
    self.target_model = DQN().to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.loss_function = nn.SmoothL1Loss()          # Huber loss: robust to outliers
    self.memory = deque(maxlen=BUFFER_SIZE)

    self.epsilon = EPSILON_START
    self.total_steps = 0
    reset_round_counters(self)

    if not os.path.isfile(STATS_FILE):
        with open(STATS_FILE, "w", newline="") as file:
            csv.writer(file).writerow(
                ["round", "steps", "score", "coins", "crates", "bombs", "kills",
                 "invalid", "killed_self", "reward", "epsilon", "loss"])
    self.logger.info("DQN training setup complete.")


def reset_round_counters(self):
    """Per-round statistics that end up as one line in training_stats.csv."""
    self.round_reward = 0.0
    self.round_losses = []
    self.round_counts = {e.COIN_COLLECTED: 0, e.CRATE_DESTROYED: 0, e.BOMB_DROPPED: 0,
                         e.KILLED_OPPONENT: 0, e.INVALID_ACTION: 0}


def count_events(self, events):
    """Add this step's events to the per-round counters."""
    for event in self.round_counts:
        self.round_counts[event] += events.count(event)


def add_custom_events(self, old_game_state, self_action, new_game_state, events):
    """Turn the difference between two game states into additional events."""
    if old_game_state is None or new_game_state is None:
        return

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    old_coin, new_coin = coin_distance(old_game_state), coin_distance(new_game_state)
    if old_coin >= 0 and new_coin >= 0:
        if new_coin < old_coin:
            events.append(MOVED_TOWARDS_COIN)
        elif new_coin > old_coin:
            events.append(MOVED_AWAY_FROM_COIN)
    elif old_coin < 0 and old_features[20] == 0:      # no coin in sight and not fleeing a bomb
        old_crate, new_crate = crate_distance(old_game_state), crate_distance(new_game_state)
        if old_crate >= 0 and new_crate >= 0:
            if new_crate < old_crate:
                events.append(MOVED_TOWARDS_CRATE)
            elif new_crate > old_crate:
                events.append(MOVED_AWAY_FROM_CRATE)

    was_in_danger = old_features[20] > 0
    is_in_danger = new_features[20] > 0
    if was_in_danger and not is_in_danger:
        events.append(ESCAPED_DANGER)
    elif was_in_danger and is_in_danger:
        events.append(STAYED_IN_DANGER)
    elif not was_in_danger and is_in_danger and self_action != 'BOMB':
        events.append(MOVED_INTO_DANGER)

    if e.BOMB_DROPPED in events:
        if old_features[23] == 0:
            events.append(SUICIDAL_BOMB)
        elif old_features[22] > 0 or old_features[24] > 0:
            events.append(GOOD_BOMB)
        else:
            events.append(USELESS_BOMB)


def reward_from_events(self, events: List[str]) -> float:
    """Sum up the rewards of all events of one step."""
    reward = sum(REWARDS.get(event, 0.0) for event in events)
    self.logger.debug(f"Awarded {reward} for events {events}")
    return reward


def optimize(self):
    """One gradient step on a random batch from the replay memory."""
    if len(self.memory) < max(LEARN_START, BATCH_SIZE):
        return None

    batch = random.sample(self.memory, BATCH_SIZE)
    states = torch.from_numpy(np.stack([t.state for t in batch]))
    actions = torch.tensor([ACTIONS.index(t.action) for t in batch], dtype=torch.int64)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)

    # states after the last step of a round are None -> their future value is 0
    non_final = torch.tensor([t.next_state is not None for t in batch])
    next_values = torch.zeros(BATCH_SIZE)
    if non_final.any():
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch if t.next_state is not None]))
        with torch.no_grad():
            next_values[non_final] = self.target_model(next_states).max(dim=1).values

    predicted = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    target = rewards + GAMMA * next_values

    loss = self.loss_function(predicted, target)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)   # keeps the updates stable
    self.optimizer.step()
    return loss.item()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Called after every step: store the transition and take a gradient step."""
    add_custom_events(self, old_game_state, self_action, new_game_state, events)
    reward = reward_from_events(self, events)

    self.round_reward += reward
    count_events(self, events)
    self.last_processed = (new_game_state["round"], new_game_state["step"])

    old_features = state_to_features(old_game_state)
    if old_features is not None:
        self.memory.append(Transition(old_features, self_action, state_to_features(new_game_state), reward))

    loss = optimize(self)
    if loss is not None:
        self.round_losses.append(loss)

    self.total_steps += 1
    if self.total_steps % TARGET_UPDATE == 0:
        self.target_model.load_state_dict(self.model.state_dict())
        self.logger.info(f"Target network updated at step {self.total_steps}.")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called once at the end of a round: final update, statistics and saving."""
    if last_game_state is None:      # cannot happen in a normal game, but keeps us safe
        return

    # The framework does not clear the event list between the last
    # game_events_occurred and this call, so an agent that survived the final
    # step receives that step's events twice. Only SURVIVED_ROUND is new here.
    if getattr(self, "last_processed", None) == (last_game_state["round"], last_game_state["step"]):
        events = [event for event in events if event == e.SURVIVED_ROUND]

    reward = reward_from_events(self, events)
    self.round_reward += reward
    count_events(self, events)

    last_features = state_to_features(last_game_state)
    if last_features is not None and last_action is not None:
        self.memory.append(Transition(last_features, last_action, None, reward))

    loss = optimize(self)
    if loss is not None:
        self.round_losses.append(loss)

    with open(STATS_FILE, "a", newline="") as file:
        csv.writer(file).writerow([
            last_game_state["round"],
            last_game_state["step"],
            last_game_state["self"][1],
            self.round_counts[e.COIN_COLLECTED],
            self.round_counts[e.CRATE_DESTROYED],
            self.round_counts[e.BOMB_DROPPED],
            self.round_counts[e.KILLED_OPPONENT],
            self.round_counts[e.INVALID_ACTION],
            int(e.KILLED_SELF in events),
            round(self.round_reward, 2),
            round(self.epsilon, 4),
            round(float(np.mean(self.round_losses)), 4) if self.round_losses else "",
        ])

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    reset_round_counters(self)

    torch.save(self.model.state_dict(), MODEL_FILE)
    if last_game_state["round"] % LOG_EVERY == 0:
        self.logger.info(f"Model saved after round {last_game_state['round']}.")
