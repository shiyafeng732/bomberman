"""
Plot the learning curve from an agent's training_stats.csv.

Usage (from the bomberman_rl folder):
    python tools/plot_training.py agent_code/my_agent/training_stats.csv
    python tools/plot_training.py agent_code/my_agent/training_stats.csv agent_code/my_agent_dqn/training_stats.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_stats(path):
    """Read the csv written by train.py into a dictionary of columns."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(rows[name]) for name in rows.dtype.names}


def smooth(values, window=50):
    """Running mean, so the noisy per-round values become readable."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    window = min(window, max(1, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main(paths):
    columns = ["coins", "crates", "reward", "killed_self"]
    titles = ["Coins per round", "Crates destroyed per round", "Total reward per round", "Suicide rate"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for path in paths:
        stats = read_stats(path)
        label = Path(path).parent.name
        for axis, column, title in zip(axes.flat, columns, titles):
            if column not in stats:
                continue
            values = smooth(stats[column])
            axis.plot(np.arange(len(values)), values, label=label)
            axis.set_title(title)
            axis.set_xlabel("round")
            axis.grid(alpha=0.3)

    axes.flat[0].legend()
    figure.suptitle("Training progress (running mean over 50 rounds)")
    figure.tight_layout()

    output = Path("results") / "training_progress.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"saved {output}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
