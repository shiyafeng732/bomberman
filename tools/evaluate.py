"""
Run a tournament-style evaluation (no training, no GUI) and print a summary table.

Usage (from the bomberman_rl folder):
    python tools/evaluate.py --agents my_agent rule_based_agent --n-rounds 100
    python tools/evaluate.py --agents my_agent --scenario coin-heaven --n-rounds 100

The numbers printed here (points per round, coins, kills, suicides, invalid
actions) are the performance metrics used in the report.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    name = f"eval_{'_vs_'.join(args.agents)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    stats_file = Path("results") / f"{name}.json"

    command = [sys.executable, "main.py", "play",
               "--agents", *args.agents,
               "--n-rounds", str(args.n_rounds),
               "--scenario", args.scenario,
               "--no-gui",
               "--save-stats", str(stats_file)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]

    print(" ".join(command))
    subprocess.run(command, check=True)

    with open(stats_file) as file:
        results = json.load(file)

    rounds = args.n_rounds
    header = f"{'agent':<24}{'points/rd':>10}{'coins/rd':>10}{'kills':>8}{'suicides':>10}{'invalid/rd':>12}"
    print("\n" + header)
    print("-" * len(header))
    for agent, stats in sorted(results["by_agent"].items(), key=lambda kv: -kv[1].get("score", 0)):
        print(f"{agent:<24}"
              f"{stats.get('score', 0) / rounds:>10.2f}"
              f"{stats.get('coins', 0) / rounds:>10.2f}"
              f"{stats.get('kills', 0):>8}"
              f"{stats.get('suicides', 0):>10}"
              f"{stats.get('invalid', 0) / rounds:>12.2f}")
    print(f"\nfull results: {stats_file}")


if __name__ == "__main__":
    main()
