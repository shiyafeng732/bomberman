"""
A/B test two test-time safety rules on top of the trained DQN, without touching
the trained weights.

  margin   only drop a bomb if the escape route leaves at least one move to
           spare.  Feature 23 only says "an escape exists", which is true even
           when the escape needs every single move, and the diagnostic shows the
           agent lives at that zero margin 69% of the time.
  mask     never pick an action the feature vector already marks as impossible
           or deadly (a move into a wall/blast, BOMB without a bomb).

Usage (from the bomberman_rl folder):
    python tools/ab_safety.py --rules base margin mask both --seeds 1 2 3
    python tools/ab_safety.py --rules base both --opponents rule_based_agent rule_based_agent rule_based_agent
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import importlib                                        # noqa: E402
import main                                             # noqa: E402
import settings as s                                    # noqa: E402

CB = None                                               # agent callbacks module
F = None                                                # its features module
ACTIONS = None


def load_agent(name):
    global CB, F, ACTIONS
    CB = importlib.import_module(f"agent_code.{name}.callbacks")
    F = importlib.import_module(f"agent_code.{name}.features")
    ACTIONS = F.ACTIONS


def bomb_margin(game_state):
    """Moves to spare if we dropped a bomb right here (-1 if we could not escape)."""
    field, pos, _, _, others, danger, free = F.parse_state(game_state)
    virtual = danger.copy()
    for (x, y) in F.blast_coords(field, pos):
        virtual[x, y] = min(virtual[x, y], s.BOMB_TIMER)
    step, distance = F.bfs_first_step(free, pos, F.safe_targets(free, virtual), danger=virtual)
    return -1 if step == -1 else s.BOMB_TIMER - distance


def make_act(rule, original_act):
    def act(self, game_state):
        if rule == "base" or (self.train and getattr(self, "epsilon", 0) > 0):
            return original_act(self, game_state)

        features = F.state_to_features(game_state)
        q = CB.q_values(self, features)
        allowed = np.ones(len(ACTIONS), dtype=bool)

        if rule in ("mask", "both"):
            allowed[:4] = features[:4] > 0          # move_ok
            allowed[5] = features[21] > 0           # bomb available
            if not allowed.any():                   # nothing left, take the risk
                allowed[:] = True

        if rule in ("margin", "both") and allowed[5]:
            if bomb_margin(game_state) < 1:
                allowed[5] = False

        q = np.where(allowed, q, -np.inf)
        best = np.flatnonzero(q == q.max())
        return ACTIONS[int(np.random.choice(best))]

    return act


def run(agent, rule, opponents, scenario, n_rounds, seed, original_act):
    CB.act = make_act(rule, original_act)
    stats_file = ROOT / "results" / f"ab_{rule}_{seed}_{datetime.now().strftime('%H%M%S%f')}.json"
    stats_file.parent.mkdir(exist_ok=True)
    sys.argv = ["main.py", "play",
                "--agents", agent, *opponents,
                "--n-rounds", str(n_rounds), "--scenario", scenario,
                "--seed", str(seed), "--no-gui", "--save-stats", str(stats_file)]
    main.main()

    by_agent = json.load(open(stats_file))["by_agent"]
    ours = next(n for n in by_agent if n.startswith(agent))
    mine = by_agent[ours]
    others = [by_agent[n] for n in by_agent if n != ours]
    best_opponent = max((o.get("score", 0) / n_rounds for o in others), default=0.0)
    return {
        "points": mine.get("score", 0) / n_rounds,
        "coins": mine.get("coins", 0) / n_rounds,
        "kills": mine.get("kills", 0) / n_rounds,
        "suicides": mine.get("suicides", 0) / n_rounds,
        "invalid": mine.get("invalid", 0) / n_rounds,
        "best_opponent": best_opponent,
    }


def main_ab():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="+", default=["base", "margin", "mask", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--opponents", nargs="*", default=["random_agent"] * 3)
    parser.add_argument("--agent", default="my_agent_dqn")
    args = parser.parse_args()

    load_agent(args.agent)
    original_act = CB.act
    print(f"\n{args.agent} | {args.n_rounds} rounds, {args.scenario},"
          f" vs {', '.join(args.opponents)}\n")
    header = f"{'rule':<8}{'seed':>5}{'points':>9}{'coins':>8}{'kills':>8}{'suicides':>10}{'invalid':>9}{'best opp':>10}"
    print(header)
    print("-" * len(header))

    totals = {}
    for rule in args.rules:
        for seed in args.seeds:
            r = run(args.agent, rule, args.opponents, args.scenario, args.n_rounds, seed,
                    original_act)
            totals.setdefault(rule, []).append(r)
            print(f"{rule:<8}{seed:>5}{r['points']:>9.2f}{r['coins']:>8.2f}{r['kills']:>8.2f}"
                  f"{r['suicides']:>10.2f}{r['invalid']:>9.2f}{r['best_opponent']:>10.2f}",
                  flush=True)

    print("-" * len(header))
    for rule, runs in totals.items():
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        print(f"{rule:<8}{'mean':>5}{mean['points']:>9.2f}{mean['coins']:>8.2f}{mean['kills']:>8.2f}"
              f"{mean['suicides']:>10.2f}{mean['invalid']:>9.2f}{mean['best_opponent']:>10.2f}")
    print("\nAB_SAFETY_DONE")


if __name__ == "__main__":
    main_ab()
