"""
Training code for Model 1 (linear Q-learning).

Update rule (n-step temporal difference with a max-bootstrap, i.e. Q-learning):

    G      = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1} + gamma^n * max_a Q(s_{t+n}, a)
    w_a   += alpha * (G - Q(s_t, a_t)) * phi(s_t)

Because Q is linear in the features, the gradient of the squared TD error with
respect to w_a is simply the feature vector -- that is the whole update.
"""

import csv
import json
import os
import pickle
from collections import deque, namedtuple
from typing import List

import numpy as np

import events as e
from .callbacks import MODEL_FILE, q_values
from .features import ACTIONS, coin_distance, crate_distance, state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ---------------------------------------------------------------- hyperparameters
ALPHA = 0.01           # learning rate
GAMMA = 0.90           # discount factor
# How many steps we look ahead before bootstrapping.
# Keep this at 1 unless you know what you are doing: with an epsilon-greedy
# behaviour policy the n-step return is off-policy for n > 1, so an action gets
# credited with the rewards of the *random* actions that happened to follow it.
# In our experiment (400 rounds, coin-heaven) n = 3 looked identical to n = 1 in
# the training curve, but the greedy agent got stuck in a WAIT loop afterwards
# (4.1 instead of 50.0 coins per round).
N_STEP = 1

EPSILON_START = float(os.environ.get("EPSILON_START", 1.0))
EPSILON_END = 0.05     # exploration we keep forever (to stay adaptive)
EPSILON_DECAY = 0.995  # multiplied with epsilon after every round

LOG_EVERY = 100        # write a "model saved" line to the log every ... rounds
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
    # rewards that mirror the real game
    e.COIN_COLLECTED: 5.0,
    e.KILLED_OPPONENT: 30.0,
    e.CRATE_DESTROYED: 1.0,
    e.COIN_FOUND: 0.5,
    e.SURVIVED_ROUND: 3.0,
    e.KILLED_SELF: -20.0,     # fires together with GOT_KILLED, so a suicide costs -50 in total
    e.GOT_KILLED: -30.0,
    e.INVALID_ACTION: -3.0,
    e.WAITED: -1.0,
    # auxiliary rewards of our own (see report: they make the signal dense)
    MOVED_TOWARDS_COIN: 1.0,
    MOVED_AWAY_FROM_COIN: -1.2,
    MOVED_TOWARDS_CRATE: 0.3,
    MOVED_AWAY_FROM_CRATE: -0.4,
    ESCAPED_DANGER: 3.0,
    STAYED_IN_DANGER: -1.0,
    MOVED_INTO_DANGER: -4.0,
    GOOD_BOMB: 5.0,
    USELESS_BOMB: -1.0,
    SUICIDAL_BOMB: -5.0,
}

# Controlled experiments override single reward values through the environment,
# so a sweep changes exactly one number and never edits this file:
#   REWARD_OVERRIDES='{"GOOD_BOMB": 3}' python main.py play ...
_raw_overrides = os.environ.get("REWARD_OVERRIDES", "").strip()
if _raw_overrides:
    try:
        _overrides = json.loads(_raw_overrides)
    except json.JSONDecodeError as _error:
        raise ValueError(f"REWARD_OVERRIDES is not valid JSON: {_raw_overrides!r}") from _error
    for _event, _value in _overrides.items():
        if _event not in REWARDS:
            raise KeyError(f"REWARD_OVERRIDES names {_event!r}, which is not one of {sorted(REWARDS)}")
        REWARDS[_event] = float(_value)
    print(f"[{__name__}] REWARD_OVERRIDES active: {_overrides}")


def setup_training(self):
    """Called once after setup() when the agent runs in training mode."""
    self.transitions = deque(maxlen=N_STEP)
    self.epsilon = EPSILON_START
    reset_round_counters(self)

    if not os.path.isfile(STATS_FILE):
        with open(STATS_FILE, "w", newline="") as file:
            csv.writer(file).writerow(
                ["round", "steps", "score", "coins", "crates", "bombs", "kills",
                 "invalid", "killed_self", "reward", "epsilon"])
    self.logger.info("Training setup complete.")


def reset_round_counters(self):
    """Per-round statistics that end up as one line in training_stats.csv."""
    self.round_reward = 0.0
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

    # --- did we get closer to a coin (or, if none is visible, to a crate)?
    old_coin, new_coin = coin_distance(old_game_state), coin_distance(new_game_state)

    fleeing = old_features[20] > 0

    if not fleeing and old_coin >= 0 and new_coin >= 0:
        if new_coin < old_coin:
            events.append(MOVED_TOWARDS_COIN)
        elif new_coin > old_coin:
            events.append(MOVED_AWAY_FROM_COIN)
    elif not fleeing and old_coin < 0:
        old_crate, new_crate = crate_distance(old_game_state), crate_distance(new_game_state)
        if old_crate >= 0 and new_crate >= 0:
            if new_crate < old_crate:
                events.append(MOVED_TOWARDS_CRATE)
            elif new_crate > old_crate:
                events.append(MOVED_AWAY_FROM_CRATE)

    # --- did we deal with bombs well? (feature 20 = "I stand inside a blast radius")
    was_in_danger = old_features[20] > 0
    is_in_danger = new_features[20] > 0
    if was_in_danger and not is_in_danger:
        events.append(ESCAPED_DANGER)
    elif was_in_danger and is_in_danger:
        events.append(STAYED_IN_DANGER)
    elif not was_in_danger and is_in_danger and self_action != 'BOMB':
        events.append(MOVED_INTO_DANGER)

    # --- was dropping the bomb a good idea?
    # features 22 / 24 = the bomb would hit a crate / an opponent, 23 = we could still escape
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


def update_model(self, transitions):
    """One n-step temporal difference update on the oldest transition of the list."""
    first = transitions[0]
    if first.state is None or first.action is None:
        return

    # discounted sum of the rewards we actually observed
    n_return = 0.0
    for k, transition in enumerate(transitions):
        n_return += (GAMMA ** k) * transition.reward

    # bootstrap with our own estimate, unless the episode ended
    last = transitions[-1]
    if last.next_state is not None:
        n_return += (GAMMA ** len(transitions)) * np.max(q_values(self, last.next_state))

    action_index = ACTIONS.index(first.action)
    td_error = n_return - self.model[action_index] @ first.state
    self.model[action_index] += ALPHA * td_error * first.state


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Called after every step: collect the transition and learn from it."""
    add_custom_events(self, old_game_state, self_action, new_game_state, events)
    reward = reward_from_events(self, events)

    self.round_reward += reward
    count_events(self, events)
    self.last_processed = (new_game_state["round"], new_game_state["step"])

    self.transitions.append(Transition(
        state_to_features(old_game_state),
        self_action,
        state_to_features(new_game_state),
        reward))

    if len(self.transitions) == N_STEP:
        update_model(self, list(self.transitions))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called once at the end of a round: final updates, statistics and saving."""
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

    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))

    # learn from every transition that is still in the buffer
    remaining = list(self.transitions)
    while remaining:
        update_model(self, remaining)
        remaining.pop(0)
    self.transitions.clear()

    # one line of statistics per round, so we can plot the learning curve later
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
        ])

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    reset_round_counters(self)

    # The model is tiny (6 x 26 numbers), so we simply store it after every round.
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)
    if last_game_state["round"] % LOG_EVERY == 0:
        self.logger.info(f"Model saved after round {last_game_state['round']}.")
