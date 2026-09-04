# Working guide for our two Bomberman agents

Everything here is run from the `bomberman_rl` folder with the conda environment
active:

```
conda activate ml_homework
```

(That environment already contains numpy, scipy, scikit-learn, matplotlib,
pytorch, pygame and tqdm. The tournament Docker image contains them too, so
`requirements.txt` is only documentation.)

## The two models

| | `agent_code/my_agent` | `agent_code/my_agent_dqn` |
|---|---|---|
| model | Q(s,a) = w_a · φ(s), one weight vector per action | MLP 26 → 128 → 128 → 6 |
| learning | n-step temporal difference Q-learning | DQN: replay memory + target network |
| lecture link | Reinforcement learning + linear regression | Neural networks / backpropagation |
| file with weights | `my-saved-model.pt` (pickle) | `dqn-model.pt` (torch state dict) |

Both use the **same** feature vector (`features.py`, 26 dimensions), so any
difference in playing strength comes from the model and not from the features.

## The curriculum (tasks 1-4 of the project sheet)

```
# Task 1 - collect revealed coins, no crates, no bombs needed
python main.py play --no-gui --agents my_agent --train 1 --scenario coin-heaven --n-rounds 2000

# Task 2 - crates: learn to bomb and to escape the bomb
python main.py play --no-gui --agents my_agent --train 1 --scenario loot-crate --n-rounds 3000
python main.py play --no-gui --agents my_agent --train 1 --scenario classic    --n-rounds 3000

# Task 3 - hunt weak opponents
python main.py play --no-gui --agents my_agent peaceful_agent coin_collector_agent --train 1 --n-rounds 3000

# Task 4 - fight the rule based agent
python main.py play --no-gui --agents my_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --n-rounds 5000
```

Each command is one line. Do not split it with a trailing `\` in PowerShell -
that is bash syntax and PowerShell replies `MissingExpressionAfterOperator`.
Its continuation character is a backtick.

Training continues from the stored weights: `setup()` loads the file whenever it
exists, so the stages build on each other. Delete `my-saved-model.pt` /
`dqn-model.pt` to start over. Replace `my_agent` by `my_agent_dqn` for model 2.

Copy the weights file after every stage, because later stages overwrite earlier
skills (finding 4 below):

```
copy agent_code\my_agent\my-saved-model.pt agent_code\my_agent\model_stage1.pt
```

Those checkpoints are what let you report a per-stage comparison table instead
of a single final number.

When you continue an already trained agent in a later stage, lower
`EPSILON_START` in `train.py` (e.g. to 0.3), otherwise the first few hundred
rounds are played almost randomly again and overwrite what the agent knew.

## Evaluating (this produces the numbers for the report)

```
python tools/evaluate.py --agents my_agent --scenario coin-heaven --n-rounds 100
python tools/evaluate.py --agents my_agent rule_based_agent --n-rounds 100
python tools/evaluate.py --agents my_agent my_agent_dqn rule_based_agent rule_based_agent --n-rounds 100
```

Evaluation always runs **without** `--train`, i.e. with epsilon = 0 and without
auxiliary rewards - exactly like the tournament.

## Plotting the learning curves

```
python tools/plot_training.py agent_code/my_agent/training_stats.csv agent_code/my_agent_dqn/training_stats.csv
```

`training_stats.csv` gets one line per round (steps, score, coins, invalid
actions, suicides, total reward, epsilon, and for the DQN the mean loss).
Rename or delete it before an experiment you want to plot separately.

## Watching a game

```
python main.py play --agents my_agent rule_based_agent --n-rounds 3
```

## Results already in hand (all of these belong in the report)

**1. Always evaluate greedily, never trust the training curve.**
With `N_STEP = 3` and `N_STEP = 1` the training curves on coin-heaven were
indistinguishable, but the greedy agents were worlds apart:

| n | coins/round training (eps > 0) | coins/round greedy (eps = 0) |
|---|---|---|
| 3 | 45.2 | 4.1 |
| 1 | 43.5 | 50.0 (all coins) |

**2. Why n = 3 failed.** With an epsilon-greedy behaviour policy the n-step
return is off-policy: a `WAIT` at step t is credited with the rewards of the
*random* actions at t+1 and t+2. Q(WAIT) is inflated and the greedy agent
freezes in a `WAIT` loop. Fixes: n = 1 (what we do), importance weighting, or
cutting the return at the first exploratory action (Watkins' Q(lambda)).

**3. The shaping made a good bomb cost more than it paid.** With the first
reward table the greedy agent picked `BOMB` zero times in 12000 actions. A bomb
destroying two crates scored +2 (GOOD_BOMB) -6 (three steps of
STAYED_IN_DANGER) -1.2 (MOVED_AWAY_FROM_CRATE while escaping) +3 (ESCAPED) +2
(crates) = **-0.2**, with a -50 risk attached. Not bombing was correct. After
raising `GOOD_BOMB` to 10, softening `STAYED_IN_DANGER` to -1 and suppressing
the crate-distance rewards while fleeing:

| reward table | greedy coins/round on loot-crate | greedy BOMB actions |
|---|---|---|
| original | 0.00 | 0 |
| rebalanced | 13.87 | 495 |

Lesson: when the agent refuses an action, add up the shaped reward over the
whole sequence that action commits it to, not just the step it happens on.

**4. Sequential curriculum stages cause catastrophic forgetting.** After 1200
rounds of loot-crate the same agent dropped from 50.0 to 4.85 coins per round on
coin-heaven, and its greedy policy there is a 2-tile LEFT/RIGHT loop (996 vs 986
actions). The crate training overwrote the coin-following weights. Worth trying:
interleave the scenarios instead of running them in sequence, lower `ALPHA` in
later stages, and test whether the DQN's replay buffer resists this better --
that is a real hypothesis your report can settle with an experiment.

## Where to tune things

* features: `features.py` (docstring at the top explains every index)
* rewards: `REWARDS` dictionary at the top of `train.py`
* hyperparameters: block marked `hyperparameters` at the top of `train.py`
* exploration: `EXPLORATION_PROBS` in `callbacks.py`
