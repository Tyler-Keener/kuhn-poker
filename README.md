# Self-Play Reinforcement Learning in Kuhn Poker

CSCE 642: Deep Reinforcement Learning, Texas A&M. Tyler Keener and Aaron Xue

**Main question:** How far from equilibrium do naive self-play Q-learning and DQN end up in
Kuhn poker, and how much does averaging past policies (fictitious play, as in Neural
Fictitious Self-Play) close the gap?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: use WSL
pip install -r requirements.txt
pytest                             # 14 sanity checks, about a minute
```

OpenSpiel ships prebuilt wheels for Linux and macOS. On Windows, use WSL.

## Running everything

```bash
python run_all.py --quick          # tiny version of every step, about 2 minutes
python run_all.py                  # full runs, 10 seeds each
```

The full run uses all but one CPU core, one seed per core. The tabular methods take
a few minutes; the neural methods (2M hands per seed) take much longer, so you can
run them separately:

```bash
python run_all.py --only naive_tabular_q tabular_fsp
python -m experiments.self_play --method nfsp --seeds 10
python -m experiments.self_play --method naive_dqn --seeds 10
python -m experiments.make_figures
```

Figures and a summary table (final exploitability with 95% confidence intervals and
Welch's t-tests) are written to `results/figures/`.

## The experiment

Two agents of the same method train against each other, and we measure the
**exploitability** of the pair's strategy: how many chips per hand a perfect opponent
would win against it. Nash equilibrium is 0. It is computed exactly from the game
tree, so the curves contain no measurement noise. The methods form a 2 x 2 design:

|                | no averaging       | averaging (fictitious play) |
|----------------|--------------------|-----------------------------|
| tabular        | `naive_tabular_q`  | `tabular_fsp`               |
| neural network | `naive_dqn`        | `nfsp`                      |

For every method we record the exploitability of the **current** policy (greedy for
value-based learners), of the **average** policy (for naive learners, the frequency of
the actions they actually played, measured after the fact), and how many information
states' greedy action **flipped** since the last checkpoint (a sign of cycling).

**Reference lines on the plots:**
- **1/6 ≈ 0.167:** the lowest exploitability any *deterministic* strategy can have in
  Kuhn poker (we checked all 4,096 of them; see `test_deterministic_policies_cannot_beat_one_sixth`).
  The equilibrium requires mixing, so a greedy learner can never get below this.
- **0.458:** the uniform random policy.
- **CFR / CFR+:** OpenSpiel's solvers, plotted against their own iterations
  (`cfr_reference.png`), because they traverse the whole tree each iteration.

## Project layout

```
kuhn_rl/
  agents/
    fixed_policy.py                  always bet / always pass / uniform random opponents
    tabular_q_learning.py            tabular Q-learning
    tabular_fictitious_self_play.py  tabular version of NFSP (Q best response + count table)
    dqn.py                           DQN: replay buffer, target network, optional Double DQN
    nfsp.py                          NFSP: DQN best response + average-policy network
    neural_components.py             network, replay buffer, reservoir buffer
    policy_helpers.py                agent interface notes, greedy policy, action counts
  game_loop.py                       plays one hand, OpenSpiel as the rules engine
  evaluation.py                      exploitability, 1/6 floor, policy flips, CFR curve
  plotting.py                        mean over seeds with 95% t-intervals
experiments/
  exp1_q_learning_vs_fixed_opponents.py   Plan step 1
  methods.py                              the four methods and all hyperparameters
  self_play.py                            Plan steps 2-4, one method, many seeds
  cfr_baseline.py                         CFR / CFR+ reference curves
  make_figures.py                         figures + summary table + t-tests
  hyperparameter_sweep.py                 try values of one setting, log to CSV
run_all.py                                runs everything
HYPERPARAMETERS.md                        how every value was chosen (keep updating)
tests/                                    pytest checks
results/                                  generated outputs (not committed)
```

Useful options: `--set name=value` overrides any hyperparameter (for example
`--set use_double_dqn=True` or `--set loss_function=huber`), and `--game leduc_poker`
runs the stretch goal. Every run saves its settings next to its results as
`*_config.json`.

## Agent interface

```python
start_hand()
act(info_state, legal_actions, info_state_tensor) -> action   # 0 = pass, 1 = bet
end_hand(final_reward)                                         # reward in [-2, +2]
action_probabilities_for(info_state, info_state_tensor, legal_actions)          # current policy
average_action_probabilities_for(info_state, info_state_tensor, legal_actions)  # average policy
```

## Code from other sources

- **OpenSpiel** (Lanctot et al., 2019), used as a library, not modified: the Kuhn and
  Leduc poker rules (`pyspiel`), `exploitability.exploitability`,
  `best_response.BestResponsePolicy`, `expected_game_score.policy_value`,
  `policy.TabularPolicy`, and the `cfr.CFRSolver` / `cfr.CFRPlusSolver` baselines.
- **PyTorch**, **NumPy**, **SciPy**, **Matplotlib**, used as libraries.
- Everything else (all learning agents, buffers, training loops, metrics built on
  top of OpenSpiel, experiments, and plots) was written for this project. We did not
  use OpenSpiel's built-in DQN or NFSP agents.
- AI assistance: the code was written with the help of Claude (Anthropic).

## Main references for the methods

Watkins & Dayan (1992), Q-learning; Mnih et al. (2015), DQN; van Hasselt, Guez &
Silver (2016), Double DQN; Brown (1951) and Robinson (1951), fictitious play;
Heinrich, Lanctot & Silver (2015), fictitious self-play; Heinrich & Silver (2016), NFSP;
Zinkevich et al. (2007), CFR; Tammelin (2014), CFR+; Kuhn (1950), Kuhn poker.
Full references are in the report.
