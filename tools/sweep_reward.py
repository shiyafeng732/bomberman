"""
Controlled experiment over one reward parameter.

Everything except the swept parameter is held fixed: the same starting weights,
the same scenario, the same number of training rounds, the same exploration
schedule and the same evaluation seed. That is what makes the resulting table a
controlled comparison rather than four unrelated training runs.

Each configuration
  1. starts from --init (e.g. model_stage1.pt), never from the previous config,
  2. trains --train-rounds rounds with the one reward value overridden,
  3. is then evaluated greedily (epsilon = 0) for --eval-rounds rounds.

Artefacts land in sweep/<param>_<value>/ so nothing is overwritten, and the
agent's own weights and training log are restored when the sweep ends.

Usage (from the bomberman_rl folder):
    python tools/sweep_reward.py --agent my_agent --param GOOD_BOMB --values 2 3 5 10
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

LIVE_MODEL = {"my_agent": "my-saved-model.pt", "my_agent_dqn": "dqn-model.pt"}


def run(command, env_extra=None):
    import os
    env = dict(os.environ)
    env.update(env_extra or {})
    subprocess.run(command, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=sorted(LIVE_MODEL))
    parser.add_argument("--param", default="GOOD_BOMB", help="event name in the REWARDS table")
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--init", default="model_stage1.pt",
                        help="checkpoint every configuration starts from")
    parser.add_argument("--scenario", default="loot-crate")
    parser.add_argument("--train-rounds", type=int, default=1500)
    parser.add_argument("--eval-rounds", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=0.3,
                        help="we continue from trained weights, so start below 1.0")
    parser.add_argument("--eval-seed", type=int, default=1,
                        help="same board sequence for every configuration")
    parser.add_argument("--eval-opponents", nargs="*", default=[])
    args = parser.parse_args()

    folder = Path("agent_code") / args.agent
    live = folder / LIVE_MODEL[args.agent]
    stats_csv = folder / "training_stats.csv"
    init = folder / args.init
    if not init.exists():
        print(f"starting checkpoint not found: {init}")
        return

    out_root = Path("sweep") / f"{args.agent}_{args.param}"
    out_root.mkdir(parents=True, exist_ok=True)

    # keep the user's current weights and training log safe
    backups = {}
    for path in (live, stats_csv):
        if path.exists():
            backups[path] = path.with_suffix(path.suffix + ".sweep-backup")
            shutil.copy(path, backups[path])

    rows = []
    try:
        for value in args.values:
            tag = f"{args.param}_{value:g}"
            case = out_root / tag
            case.mkdir(exist_ok=True)
            print(f"\n=== {tag} : training {args.train_rounds} rounds on {args.scenario} ===")
            started = time.time()

            # every configuration starts from the same weights and an empty log
            shutil.copy(init, live)
            if stats_csv.exists():
                stats_csv.unlink()

            run([sys.executable, "main.py", "play", "--no-gui", "--train", "1",
                 "--agents", args.agent, "--scenario", args.scenario,
                 "--n-rounds", str(args.train_rounds)],
                env_extra={"REWARD_OVERRIDES": json.dumps({args.param: value}),
                           "EPSILON_START": str(args.epsilon_start)})

            shutil.copy(live, case / "model.pt")
            if stats_csv.exists():
                shutil.copy(stats_csv, case / "training_stats.csv")

            print(f"    trained in {(time.time() - started) / 60:.1f} min, evaluating greedily")
            eval_json = case / "eval.json"
            run([sys.executable, "main.py", "play", "--no-gui",
                 "--agents", args.agent, *args.eval_opponents,
                 "--scenario", args.scenario, "--n-rounds", str(args.eval_rounds),
                 "--seed", str(args.eval_seed), "--save-stats", str(eval_json)])

            stats = json.load(open(eval_json))["by_agent"][args.agent]
            n = args.eval_rounds
            rows.append({
                "value": value,
                "score": stats.get("score", 0) / n,
                "coins": stats.get("coins", 0) / n,
                "crates": stats.get("crates", 0) / n,
                "bombs": stats.get("bombs", 0) / n,
                "crates_per_bomb": (stats.get("crates", 0) / stats["bombs"]) if stats.get("bombs") else 0.0,
                "suicides": stats.get("suicides", 0) / n,
                "steps": stats.get("steps", 0) / n,
            })
    finally:
        for path, backup in backups.items():
            shutil.copy(backup, path)
            backup.unlink()
        print("\n(the agent's own weights and training log have been restored)")

    if not rows:
        return

    # ---- report-ready output
    columns = [("value", args.param, "{:.0f}"), ("score", "score/rd", "{:.2f}"),
               ("coins", "coins/rd", "{:.2f}"), ("crates", "crates/rd", "{:.1f}"),
               ("bombs", "bombs/rd", "{:.1f}"), ("crates_per_bomb", "crates/bomb", "{:.2f}"),
               ("suicides", "suicides/rd", "{:.2f}"), ("steps", "steps/rd", "{:.0f}")]

    print(f"\nControlled sweep of {args.param} | start {args.init} | "
          f"{args.train_rounds} training rounds on {args.scenario} | "
          f"{args.eval_rounds} greedy rounds, seed {args.eval_seed}\n")
    print("| " + " | ".join(head for _, head, _ in columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(fmt.format(row[key]) for key, _, fmt in columns) + " |")

    best = max(rows, key=lambda r: r["score"])
    print(f"\nbest score/round: {args.param} = {best['value']:g} ({best['score']:.2f})")

    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as file:
        import csv as csv_module
        writer = csv_module.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"table also written to {csv_path}")


if __name__ == "__main__":
    main()
