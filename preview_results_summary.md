# Results summary (kuhn_poker)

Final = each seed's mean over the last 10% of checkpoints; shown as mean [95% confidence interval] across seeds.

| Method | Seeds | Hands | Current policy | Average policy | Policy flips (last half) |
|---|---|---|---|---|---|
| Naive self-play, tabular Q | 10 | 500,000 | 0.2192 [0.2060, 0.2324] | 0.1178 [0.1162, 0.1194] | 1.94 |
| Fictitious self-play, tabular | 10 | 500,000 | 0.2325 [0.2129, 0.2521] | 0.0150 [0.0141, 0.0159] | 2.32 |
| Naive self-play, DQN | 3 | 1,000,000 | 0.2833 [0.2116, 0.3550] | 0.0677 [0.0524, 0.0830] | 0.65 |
| NFSP (neural fictitious self-play) | 3 | 1,000,000 | 0.3111 [0.2446, 0.3777] | 0.0250 [0.0203, 0.0297] | 0.61 |

Uniform random policy: 0.4583

Best any deterministic policy can do (1/6): 0.1667

CFR after 1000 iterations: 0.00094; CFR+: 0.00009

## Welch's t-tests on final exploitability

| Comparison | Difference in means | t | p |
|---|---|---|---|
| Naive self-play, tabular Q (current) vs. Fictitious self-play, tabular (average) | +0.2042 | 34.92 | 5.4e-11 |
| Naive self-play, tabular Q (average) vs. Fictitious self-play, tabular (average) | +0.1028 | 124.84 | 6.7e-23 |
| Naive self-play, DQN (current) vs. NFSP (neural fictitious self-play) (average) | +0.2583 | 15.47 | 0.004 |
| Naive self-play, DQN (average) vs. NFSP (neural fictitious self-play) (average) | +0.0427 | 11.49 | 0.0039 |
| Fictitious self-play, tabular (average) vs. NFSP (neural fictitious self-play) (average) | -0.0100 | -8.64 | 0.0058 |
