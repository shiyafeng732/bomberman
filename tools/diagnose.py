"""
Turn a saved match into a weakness report.

Raw counters ("we scored 864, they scored 1857") say that we are behind but not
why. This script divides the counters into *efficiency ratios* that isolate the
individual skills, and compares each one against the opponent, so the weakest
skill becomes visible.

Usage (from the bomberman_rl folder):
    python tools/diagnose.py results/eval_my_agent_vs_rule_based_agent_*.json
    python tools/diagnose.py --agent my_agent --opponents rule_based_agent --n-rounds 300
"""

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# name -> (numerator, denominator, "higher is better?", what the number means)
RATIOS = [
    ("score / round",      "score",    "rounds",  True,  "the bottom line"),
    ("coins / round",      "coins",    "rounds",  True,  "1 point each"),
    ("kills / round",      "kills",    "rounds",  True,  "5 points each"),
    ("crates / bomb",      "crates",   "bombs",   True,  "bomb placement quality"),
    ("coins / crate",      "coins",    "crates",  True,  "do we pick up what we uncover"),
    ("coins / 100 steps",  "coins",    "steps",   True,  "collection speed while alive"),
    ("steps / round",      "steps",    "rounds",  True,  "how long we survive"),
    ("bombs / 100 steps",  "bombs",    "steps",   None,  "how bomb-happy we are"),
    ("suicides / round",   "suicides", "rounds",  False, "self-inflicted deaths"),
    ("invalid / 100 steps","invalid",  "steps",   False, "wasted actions"),
]

SCALE = {"coins / 100 steps": 100, "bombs / 100 steps": 100, "invalid / 100 steps": 100}


def run_match(agent, opponents, scenario, n_rounds, seed):
    """Play a greedy match and return the path of the statistics file."""
    Path("results").mkdir(exist_ok=True)
    stats_file = Path("results") / f"diagnose_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    command = [sys.executable, "main.py", "play",
               "--agents", agent, *opponents,
               "--n-rounds", str(n_rounds), "--scenario", scenario,
               "--no-gui", "--save-stats", str(stats_file)]
    if seed is not None:
        command += ["--seed", str(seed)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return stats_file


def ratio(stats, numerator, denominator, scale):
    bottom = stats.get(denominator, 0)
    return None if not bottom else scale * stats.get(numerator, 0) / bottom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", nargs="?", help="a results/*.json written by --save-stats")
    parser.add_argument("--agent", default=None, help="run a fresh match for this agent instead")
    parser.add_argument("--opponents", nargs="*", default=["rule_based_agent"])
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--n-rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.stats_file:
        matches = sorted(glob.glob(args.stats_file))
        if not matches:
            print(f"no file matches {args.stats_file}")
            return
        path = Path(matches[-1])
    elif args.agent:
        path = run_match(args.agent, args.opponents, args.scenario, args.n_rounds, args.seed)
    else:
        print("give a stats file or --agent")
        return

    by_agent = json.load(open(path))["by_agent"]
    if len(by_agent) < 2:
        print("this match has only one agent, nothing to compare against")
        return

    # our agent is the one whose name is not a provided agent, else the first
    provided = ("rule_based_agent", "coin_collector_agent", "peaceful_agent", "random_agent")
    ours = next((n for n in by_agent if not n.startswith(provided)), list(by_agent)[0])
    theirs = [n for n in by_agent if n != ours]
    # average the opponents so several identical opponents count once
    opponent = {key: sum(by_agent[n].get(key, 0) for n in theirs) / len(theirs)
                for key in by_agent[ours]}

    print(f"\nsource   {path.name}")
    print(f"us       {ours}")
    print(f"them     {', '.join(theirs)}"
          f"{' (averaged)' if len(theirs) > 1 else ''}\n")

    header = f"{'metric':<21}{'us':>10}{'them':>10}{'ratio':>9}   {'':<3}what it measures"
    print(header)
    print("-" * 92)

    weaknesses = []
    for name, numerator, denominator, higher_better, meaning in RATIOS:
        scale = SCALE.get(name, 1)
        us = ratio(by_agent[ours], numerator, denominator, scale)
        them = ratio(opponent, numerator, denominator, scale)
        if us is None or them is None:
            continue
        relative = us / them if them else float("inf")

        flag = "   "
        if higher_better is not None:
            # how far behind are we, in the direction that is bad for us
            behind = (1 - relative) if higher_better else (relative - 1)
            if behind > 0.30:
                flag, _ = "<<<", weaknesses.append((behind, name, us, them))
            elif behind > 0.10:
                flag = " < "

        print(f"{name:<21}{us:>10.2f}{them:>10.2f}{relative:>8.0%}   {flag} {meaning}")

    print()
    if weaknesses:
        print("biggest gaps (>30% behind), worst first:")
        for behind, name, us, them in sorted(weaknesses, reverse=True):
            print(f"  {behind:>5.0%}  {name:<20} {us:.2f} against {them:.2f}")
    else:
        print("no metric is more than 30% behind the opponent")
    print()


if __name__ == "__main__":
    main()
