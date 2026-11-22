# How each hyperparameter was chosen

The final report must say, for every parameter, how its value was reached (first
try, trial and error, roughly how many trials). This file is that record. Add a
line every time you try something, including what didn't work. The raw numbers
from `experiments/hyperparameter_sweep.py` are appended to
`results/sweeps/sweep_log.csv`, but `results/` is not committed, so copy the
conclusions here.

Tuning runs use seeds 1000 and up; the final runs use seeds 0-9, so the reported
results never reuse the seeds that were used for tuning.

## Current defaults (`experiments/methods.py`)

| Setting | Value | Used by | How it was chosen |
|---|---|---|---|
| Q-learning rate | 0.01 | tabular Q, tabular FSP | Trial and error, 2 values (see log) |
| Starting / minimum epsilon | 0.3 / 0.02 | all | First choice, not tuned |
| Epsilon decay per hand | 0.99995 | all | First choice: reaches the minimum after about 55k hands |
| Discount factor gamma | 1.0 | all | Not tuned; hands last at most 3 decisions (see proposal) |
| Anticipatory parameter eta | 0.1 | tabular FSP, NFSP | Taken from Heinrich & Silver (2016), not tuned |
| Hidden layers | one layer of 64 | DQN, NFSP | First choice (Kuhn has only 12 information states) |
| DQN learning rate (Adam) | 0.001 | DQN, NFSP | First choice; see log for attempts |
| Batch size | 128 | DQN, NFSP | First choice |
| Replay buffer | 100,000 transitions | DQN, NFSP | First choice |
| Learn every | 16 decisions | DQN, NFSP | First choice; see log |
| Target network update | every 250 gradient steps | DQN, NFSP | First choice |
| Loss function | mean squared error | DQN, NFSP | Changed from Huber after diagnosing a bias (see log) |
| Average-policy learning rate (Adam) | 0.001 | NFSP | First choice; checked that the network matches the exact average of its data |
| Reservoir buffer | 200,000 samples | NFSP | First choice |
| Training hands | 500k tabular, 2M neural | all | From preview runs: tabular levels off by 500k; NFSP was still improving at 1M (3 seeds), so neural methods get 2M, the same for DQN and NFSP |

## Log

**Nov 3: tabular Q-learning rate, 0.05 -> 0.01** (2 values, 3-5 seeds each, Plan step 1).
With 0.05, Q-learning against the always-bet opponent sometimes learned to fold a
Queen even though calling is better, ending at +0.22 instead of the best-response
value +0.333 chips/hand. Rewards of +/-2 make constant-step estimates noisy enough
to flip close decisions. With 0.01, all three fixed opponents were exploited
exactly (5 seeds). Kept 0.01 for every tabular method.

**Nov 14: NFSP stuck at exploitability ~0.15** (single seed, 150k-1M hands).
Diagnosis, in order:
1. The average-policy network matched the exact frequency of the actions in its
   reservoir to within 0.03-0.07 per state, so supervised learning was fine.
2. The DQN best response fell 0.1+ chips/hand short of the true best response
   to the opponent's average policy; tabular FSP's fell about 0.01 short.
3. Tried, all with the Huber loss and none of them helped: learn every 4
   decisions; learning rate 0.01; learning rate 0.0001 with learning every 4;
   batch 512 with learning rate 0.0003; 128 hidden units with target update
   every 1000 (5 trials).
4. DQN alone against a *fixed* near-equilibrium opponent (CFR after 15
   iterations) also stayed 0.06 short, while tabular Q reached 0. So the
   problem was DQN itself, not self-play.
5. Q(Queen, bet first) was -0.06 in DQN vs. about -0.4 in tabular Q. Cause: the
   Huber loss behaves like absolute error for errors above 1, so it estimates
   something between the mean and the median. Betting the Queen ends in +1 or
   -2, and the median sits far from the mean. Original DQN clipped rewards to
   [-1, 1], where this hardly matters.
6. Switched to mean squared error: Q(Queen, bet first) became -0.37, DQN reached
   the exact best response, and NFSP's exploitability fell 0.153 -> 0.103 over
   300k hands instead of staying flat.

The trials in step 3 should be re-run with mean squared error before you drop
them from consideration (use `hyperparameter_sweep.py`).

**Nov 14: NFSP learn every 4 decisions instead of 16, with mean squared error** (1 seed).
At 120k hands: exploitability 0.138 vs. 0.128 with the default, and about 2.5 times
slower. No sign of a gain, so kept 16. One seed is not conclusive.

**Nov 14: preview runs** (10 seeds tabular at 500k; 3 seeds neural at 1M).
Tabular FSP average 0.015, naive tabular Q current 0.219 / after-the-fact average 0.118.
NFSP average 0.025 and still falling at 1M; naive DQN current 0.283 / after-the-fact
average 0.068. Raised the neural budget to 2M hands for the final runs.

<!-- Add new entries below: date, what you changed, how many trials/seeds, what happened. -->

**Nov 22: final runs** (10 seeds each, 2M hands, seeds 0-9).
Tabular methods rerun at 2M hands so all four methods share one budget.
Naive tabular Q: current 0.230, after-the-fact average 0.121. Tabular FSP average 0.0128.
Naive DQN: current 0.281, after-the-fact average 0.067. NFSP average 0.0122.
Tabular FSP vs. NFSP: not significantly different (Welch's t = 1.76, p = 0.097).
Plan step 1 (fixed opponents, 100k hands): best response matched exactly in all 10 seeds.

**Nov 22: Huber-loss ablation** (NFSP, 10 seeds, 2M hands, loss_function=huber).
Average exploitability 0.147 [0.136, 0.157] vs. 0.0122 [0.0116, 0.0128] with squared error
(Welch's t = 29.1, p = 2.9e-10). Confirms the single-seed diagnosis from Nov 14.

**Nov 22: correction to the Nov 3 learning-rate entry** (experiments/diagnostics.py, 10 seeds, 100k hands).
Against the always-bet opponent, learning rate 0.05 reached the best response (+0.333) in all 10 seeds,
the same as 0.01. The earlier single-seed failure (+0.22) did not reproduce. 0.01 stays as a first choice.

**Nov 22: fixed-opponent diagnostic** (experiments/diagnostics.py, 10 seeds, 100k hands, opponent = CFR after 15 iterations).
Shortfall vs. best response: tabular 0.001, DQN squared error 0.007, DQN Huber 0.053 [0.040, 0.067].
Value of betting the Queen first: exact -0.433; tabular -0.517; squared error -0.484; Huber -0.073.
