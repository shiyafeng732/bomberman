"""
Evaluate several training checkpoints of one agent under identical conditions
and print them as a table. This is the per-stage comparison for the report.

Usage (from the bomberman_rl folder):
    python tools/compare_checkpoints.py --agent my_agent --scenario coin-heaven --n-rounds 100
    python tools/compare_checkpoints.py --agent my_agent --scenario classic --n-rounds 100 --opponents rule_based_agent

By default every model_stage*.pt in the agent folder is evaluated, in order.
Each checkpoint is temporarily copied over the agent's live weights file and the
original is put back afterwards, so nothing is lost.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# the file each agent actually loads in setup()
LIVE_MODEL = {"my_agent": "my-saved-model.pt", "my_agent_dqn": "dqn-model.pt"}


def run_match(agent, opponents, scenario, n_rounds, seed):
    """Play one greedy match and return the statistics of `agent`."""
    stats_file = Path("results") / f"cmp_{datetime.now().strftime('%H%M%S%f')}.json"
    command = [sys.executable, "main.py", "play",
               "--agents", agent, *opponents,
               "--n-rounds", str(n_rounds),
               "--scenario", scenario,
               "--no-gui",
               "--save-stats", str(stats_file)]
    if seed is not None:
        command += ["--seed", str(seed)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(stats_file) as file:
        results = json.load(file)
    stats_file.unlink()
    return results["by_agent"][agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=sorted(LIVE_MODEL))
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="checkpoint files; default: every model_stage*.pt in the agent folder")
    args = parser.parse_args()

    folder = Path("agent_code") / args.agent
    live = folder / LIVE_MODEL[args.agent]
    checkpoints = ([Path(c) for c in args.checkpoints] if args.checkpoints
                   else sorted(folder.glob("model_stage*.pt")))
    if not checkpoints:
        print(f"no checkpoints found in {folder}")
        return

    backup = live.with_suffix(".pt.backup")
    if live.exists():
        shutil.copy(live, backup)

    header = (f"{'checkpoint':<22}{'points/rd':>11}{'coins/rd':>10}{'kills/rd':>10}"
              f"{'suicide rate':>14}{'invalid/rd':>12}")
    print(f"\n{args.agent} | scenario {args.scenario} | {args.n_rounds} rounds "
          f"| opponents: {', '.join(args.opponents) or 'none'}")
    print(header)
    print("-" * len(header))

    try:
        for checkpoint in checkpoints:
            shutil.copy(checkpoint, live)
            stats = run_match(args.agent, args.opponents, args.scenario, args.n_rounds, args.seed)
            rounds = args.n_rounds
            print(f"{checkpoint.name:<22}"
                  f"{stats.get('score', 0) / rounds:>11.2f}"
                  f"{stats.get('coins', 0) / rounds:>10.2f}"
                  f"{stats.get('kills', 0) / rounds:>10.2f}"
                  f"{stats.get('suicides', 0) / rounds:>14.2f}"
                  f"{stats.get('invalid', 0) / rounds:>12.2f}")
    finally:
        if backup.exists():
            shutil.copy(backup, live)
            backup.unlink()
        print("\n(original weights restored)")


if __name__ == "__main__":
    main()
