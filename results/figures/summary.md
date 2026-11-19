# Results summary (kuhn_poker)

Final = each seed's mean over the last 10% of checkpoints; shown as mean [95% confidence interval] across seeds.

| Method | Seeds | Hands | Current policy | Average policy | Policy flips (last half) |
|---|---|---|---|---|---|
| Naive self-play, tabular Q | 10 | 500,000 | 0.2192 [0.2060, 0.2324] | 0.1178 [0.1162, 0.1194] | 1.94 |
| Fictitious self-play, tabular | 10 | 500,000 | 0.2325 [0.2129, 0.2521] | 0.0150 [0.0141, 0.0159] | 2.32 |
| Naive self-play, DQN | 10 | 2,000,000 | 0.2808 [0.2724, 0.2893] | 0.0669 [0.0638, 0.0700] | 1.16 |
| NFSP (neural fictitious self-play) | 10 | 2,000,000 | 0.2300 [0.2129, 0.2471] | 0.0122 [0.0116, 0.0128] | 1.72 |

Uniform random policy: 0.4583

Best any deterministic policy can do (1/6): 0.1667

CFR after 1000 iterations: 0.00094; CFR+: 0.00009

## Welch's t-tests on final exploitability

| Comparison | Difference in means | t | p |
|---|---|---|---|
| Naive self-play, tabular Q (current) vs. Fictitious self-play, tabular (average) | +0.2042 | 34.92 | 5.4e-11 |
| Naive self-play, tabular Q (average) vs. Fictitious self-play, tabular (average) | +0.1028 | 124.84 | 6.7e-23 |
| Naive self-play, DQN (current) vs. NFSP (neural fictitious self-play) (average) | +0.2686 | 71.70 | 7.9e-14 |
| Naive self-play, DQN (average) vs. NFSP (neural fictitious self-play) (average) | +0.0547 | 39.45 | 5e-12 |
| Fictitious self-play, tabular (average) vs. NFSP (neural fictitious self-play) (average) | +0.0028 | 5.85 | 2.7e-05 |
