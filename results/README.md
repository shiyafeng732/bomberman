# Greedy evaluation results

Every `.json` file here is the output of the framework's `--save-stats` option for
one greedy test game (training off, epsilon = 0). `by_agent` holds the totals of
each agent over all rounds (score, coins, kills, suicides, invalid actions,
crates, bombs, steps, thinking time), `by_round` the per-round totals. Divide by
`rounds` to get the per-round numbers used in the report. The opponents of a game
are the other entries in `by_agent`.

All games use the `classic` scenario unless noted. E1 to E7 refer to the
experiments in the report.

## Early design experiments (linear agent, 30 Aug)

| Experiment | File | Setting |
|---|---|---|
| E1, n = 3 | `eval_my_agent_20260830-193413.json` | coin-heaven, 20 rounds |
| E1, n = 1 | `eval_my_agent_20260830-193948.json` | coin-heaven, 20 rounds |
| E2, first reward table | `eval_my_agent_20260830-200719.json` | loot-crate, 30 rounds |
| E2, rebalanced rewards | `eval_my_agent_20260830-201338.json` | loot-crate, 30 rounds |
| E4, coin skill after 1,200 loot-crate rounds | `eval_my_agent_20260830-201400.json` | coin-heaven, 20 rounds |

`eval_my_agent_20260830-194141.json` and `eval_my_agent_20260830-201406.json`
are other early tests that the report does not use. E3 (the `GOOD_BOMB` sweep)
lives in `sweep/my_agent_GOOD_BOMB/`.

## E5: development snapshots against rule_based_agent

| Agent version | Files |
|---|---|
| DQN, early September (1 vs 1, 300 rounds) | `eval_my_agent_dqn_vs_rule_based_agent_20260904-090537.json` |
| DQN stage 5 (1 vs 1, 3 x 100 rounds) | `eval_my_agent_dqn_vs_rule_based_agent_20260915-015706.json`, `...-015839.json`, `...-020006.json` |
| DQN stage 5 (1 vs 3, 100 rounds) | `eval_my_agent_dqn_vs_rule_based_agent_vs_rule_based_agent_vs_rule_based_agent_20260915-020135.json` |
| DQN during the rejected fine-tuning (1 vs 1, 3 x 100 rounds) | `eval_my_agent_dqn_vs_rule_based_agent_20260915-202419.json`, `...-202606.json`, `...-202744.json` |
| DQN during the rejected fine-tuning (1 vs 3, 100 rounds) | `eval_my_agent_dqn_vs_rule_based_agent_vs_rule_based_agent_vs_rule_based_agent_20260915-202923.json` |
| Linear, early September (1 vs 1, 300 rounds) | `eval_my_agent_vs_rule_based_agent_20260904-015138.json` |
| Linear, after the sequential curriculum (1 vs 1) | `eval_my_agent_vs_rule_based_agent_20260906-230703.json` |
| Linear, after the interleaved curriculum (1 vs 1) | `eval_my_agent_vs_rule_based_agent_20260910-100626.json` |

`eval_my_agent_vs_rule_based_agent_vs_rule_based_agent_vs_rule_based_agent_20260901-160300.json`
(linear, 1 vs 3, 60 rounds) is an early test that the report does not use.

## E6 and E7: final comparison and test-time rules (21 Sep)

Written by `tools/ab_safety.py`. The file name is `ab_<rule>_<seed>_<time>.json`,
100 rounds per file, seeds 1 to 3. Weights: DQN `model_stage5.pt`, linear
`model_stage2.pt`. E6 uses the `mask` rows.

| Agent | Opponents | Rule | Files (time stamps for seeds 1, 2, 3) |
|---|---|---|---|
| DQN | 3 x random_agent | base | 180843, 180925, 181013 |
| DQN | 3 x random_agent | margin | 181105, 181159, 181306 |
| DQN | 3 x random_agent | mask | 181435, 181536, 181636 |
| DQN | 3 x random_agent | both | 181750, 181907, 182027 |
| DQN | 3 x rule_based_agent | base | 180838, 181126, 181520 |
| DQN | 3 x rule_based_agent | margin | 181901, 182213, 182441 |
| DQN | 3 x rule_based_agent | mask | 182709, 182940, 183247 |
| DQN | 3 x rule_based_agent | both | 183651, 184012, 184323 |
| Linear | 3 x random_agent | base | 184919, 185006, 185051 |
| Linear | 3 x random_agent | mask | 185138, 185228, 185311 |
| Linear | 3 x rule_based_agent | base | 184914, 185207, 185433 |
| Linear | 3 x rule_based_agent | mask | 185638, 185837, 190106 |

`ab_base_1_180654286915.json` is an earlier repeat of the DQN base run (seed 1,
random opponents). The files with only 3 or 5 rounds (`ab_base_1_180737452767`,
`ab_both_1_180738494635`, `ab_base_1_184902136585`, `ab_mask_1_184903576991`)
are quick checks of the script. The report uses none of these.

## Other files

- `escape_classic_seed1.json` to `escape_classic_seed3.json`: DQN stage 5
  without the action mask against 3 x random_agent, 100 rounds per seed, written
  by `tools/diagnose_escape.py`. Together with the DQN `base` runs above they
  give the local self-kill rate that the report compares with the pre-run.
- `eval_my_agent_dqn_vs_random_agent_vs_random_agent_vs_random_agent_20260921-184728.json`:
  the final DQN (with the action mask) against 3 x random_agent, 100 rounds,
  no seed.
- `interleaved_curves.png`: training statistics of both agents during the
  interleaved curriculum.
