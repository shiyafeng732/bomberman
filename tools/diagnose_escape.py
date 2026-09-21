"""
Why does the DQN blow itself up?

The game statistics only say *that* the agent kills itself.  This script records
what the agent does while it is standing inside a blast radius, broken down by
the bomb timer, and reports

  * how often an escape route exists and the agent does not take it,
  * how often it wastes a step on WAIT / BOMB while a bomb is ticking,
  * how often it walks into a two-tile loop (A-B-A-B).

The interesting column is the breakdown over the timer: the feature vector does
not contain the timer, so the policy cannot become more urgent as the bomb gets
closer to going off, and the numbers below make that visible.

Usage (from the bomberman_rl folder):
    python tools/diagnose_escape.py --n-rounds 100 --seed 1
    python tools/diagnose_escape.py --n-rounds 100 --seed 1 --model model_stage5.pt
"""

import argparse
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import main                                             # noqa: E402
import settings as s                                    # noqa: E402
import agent_code.my_agent_dqn.callbacks as CB          # noqa: E402
import agent_code.my_agent_dqn.features as F            # noqa: E402

ACTIONS = F.ACTIONS
MOVES = ACTIONS[:4]

# counters, filled by the wrapper around act()
steps = 0
danger_steps = 0
escape_known = 0
escape_taken = 0
wasted_in_danger = Counter()        # action -> count, only while in danger
by_timer = defaultdict(lambda: {"n": 0, "escape_known": 0, "escape_taken": 0, "wait_or_bomb": 0,
                                "escape_flagged": 0, "no_escape": 0})
instead = defaultdict(Counter)      # timer -> what it did instead of escaping
slack = Counter()                   # (timer - distance to safety) -> count
loop_steps = 0
bombs_dropped = 0
bombs_unsafe = 0                    # bomb dropped although feature 23 says "no escape"
positions = deque(maxlen=4)


def record(game_state, action):
    global steps, danger_steps, escape_known, escape_taken, loop_steps
    global bombs_dropped, bombs_unsafe

    steps += 1
    features = F.state_to_features(game_state)
    field, pos, _, _, _, danger, _ = F.parse_state(game_state)
    chosen = ACTIONS.index(action)

    positions.append(pos)
    if len(positions) == 4 and positions[0] == positions[2] and positions[1] == positions[3] \
            and positions[0] != positions[1]:
        loop_steps += 1

    if action == 'BOMB' and features[21] > 0:
        bombs_dropped += 1
        if features[23] == 0:
            bombs_unsafe += 1

    if features[20] == 0:
        return

    timer = int(danger[pos])
    bucket = by_timer[timer]
    danger_steps += 1
    bucket["n"] += 1

    # how much margin is left: we need `distance` moves to reach safety and have
    # `timer` moves before the blast. slack < 0 means the round is already lost.
    _, distance = F.bfs_first_step(F.build_free_map(field, game_state['bombs'],
                                                    [xy for (_, _, _, xy) in game_state['others']]),
                                   pos, F.safe_targets(
                                       F.build_free_map(field, game_state['bombs'],
                                                        [xy for (_, _, _, xy) in game_state['others']]),
                                       danger), danger=danger)
    if distance >= 0:
        slack[timer - distance] += 1
    else:
        bucket["no_escape"] += 1        # nothing reachable in time, the round is lost

    arrows = np.flatnonzero(features[12:16] > 0)
    if arrows.size:
        escape_known += 1
        bucket["escape_known"] += 1
        if chosen == arrows[0]:
            escape_taken += 1
            bucket["escape_taken"] += 1
        else:
            instead[timer][action] += 1
            # is the tile that would save us itself flagged as "about to explode"?
            if features[16 + arrows[0]] > 0:
                bucket["escape_flagged"] += 1

    if action in ('WAIT', 'BOMB'):
        wasted_in_danger[action] += 1
        bucket["wait_or_bomb"] += 1


def main_diagnose():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--opponents", nargs="*", default=["random_agent"] * 3)
    parser.add_argument("--model", default=None, help="weights file inside the agent folder")
    args = parser.parse_args()

    if args.model:
        CB.MODEL_FILE = str(ROOT / "agent_code" / "my_agent_dqn" / args.model)

    original_act = CB.act

    def wrapped_act(self, game_state):
        action = original_act(self, game_state)
        try:
            record(game_state, action)
        except Exception as error:                      # never break the match
            print(f"record failed: {error}")
        return action

    CB.act = wrapped_act

    stats_file = ROOT / "results" / f"escape_{args.scenario}_seed{args.seed}.json"
    stats_file.parent.mkdir(exist_ok=True)
    sys.argv = ["main.py", "play",
                "--agents", "my_agent_dqn", *args.opponents,
                "--n-rounds", str(args.n_rounds), "--scenario", args.scenario,
                "--seed", str(args.seed), "--no-gui",
                "--save-stats", str(stats_file)]
    main.main()

    import json
    by_agent = json.load(open(stats_file))["by_agent"]
    ours = next(n for n in by_agent if n.startswith("my_agent_dqn"))
    mine = by_agent[ours]
    rounds = args.n_rounds

    print(f"\nmodel        {Path(CB.MODEL_FILE).name}")
    print(f"match        {rounds} rounds, {args.scenario}, vs {', '.join(args.opponents)}\n")
    print(f"score/round        {mine.get('score', 0) / rounds:.2f}")
    print(f"suicides/round     {mine.get('suicides', 0) / rounds:.2f}"
          f"   ({mine.get('suicides', 0)} of {rounds} rounds)")
    print(f"steps              {steps}   of which in a blast: {danger_steps}"
          f" ({100 * danger_steps / max(steps, 1):.0f}%)")
    print(f"bombs dropped      {bombs_dropped}"
          f"   without an escape route (f23=0): {bombs_unsafe}")
    print(f"A-B-A-B loop steps {loop_steps} ({100 * loop_steps / max(steps, 1):.0f}% of all steps)\n")

    print("while standing in a blast:")
    print(f"  escape route known   {escape_known} steps")
    print(f"  escape step taken    {escape_taken} ({100 * escape_taken / max(escape_known, 1):.0f}%)")
    for action, count in wasted_in_danger.most_common():
        print(f"  chose {action:<5}          {count} ({100 * count / max(danger_steps, 1):.0f}% of danger steps)")

    print("\nbroken down by the bomb timer on our own tile")
    print("(timer 4 = four moves left, timer 0 = the blast is on us now)\n")
    print(f"{'timer':>6}{'steps':>8}{'escape known':>15}{'escape taken':>15}{'WAIT/BOMB':>12}"
          f"{'missed & flagged':>18}")
    print("-" * 74)
    for timer in sorted(by_timer):
        bucket = by_timer[timer]
        known, taken = bucket["escape_known"], bucket["escape_taken"]
        missed = known - taken
        rate = f"{100 * taken / known:.0f}%" if known else "-"
        flag = f"{bucket['escape_flagged']}/{missed}" if missed else "-"
        print(f"{timer:>6}{bucket['n']:>8}{known:>15}{rate:>15}{bucket['wait_or_bomb']:>12}{flag:>18}")
    print("\n'missed & flagged' = of the steps where it ignored the escape arrow, how many had")
    print("feature 16-19 raised on exactly that tile (the 'neighbour explodes soon' flag).\n")

    # A tile with timer t still allows t+1 moves, so reaching safety needs
    # distance <= timer + 1, i.e. slack >= -1. slack = -1 therefore means
    # "every remaining move has to be the right one".
    total_slack = sum(slack.values())
    no_escape = sum(b["no_escape"] for b in by_timer.values())
    print("spare moves while standing in a blast"
          " (timer + 1 - distance to the nearest safe tile):")
    for value in sorted(slack):
        share = 100 * slack[value] / max(total_slack, 1)
        label = "   <- no move may be wasted" if value + 1 == 0 else ""
        print(f"  {value + 1:>2} spare: {slack[value]:>6} steps ({share:4.1f}%){label}")
    print(f"  no escape at all: {no_escape} steps"
          f" ({100 * no_escape / max(danger_steps, 1):.1f}% of danger steps, the round is lost)\n")

    print("what it did instead of escaping:")
    for timer in sorted(instead):
        summary = "  ".join(f"{a} {c}" for a, c in instead[timer].most_common())
        print(f"  timer {timer}: {summary}")
    print()


if __name__ == "__main__":
    main_diagnose()
