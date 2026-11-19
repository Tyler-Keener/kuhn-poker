# Results summary (kuhn_poker)

Final = each seed's mean over the last 10% of checkpoints; shown as mean [95% confidence interval] across seeds.

| Method | Seeds | Hands | Current policy | Average policy | Policy flips (last half) |
|---|---|---|---|---|---|
| Naive self-play, tabular Q | 2 | 5,000 | 0.2792 [-0.7267, 1.2851] | 0.1191 [-0.2137, 0.4519] | 0.74 |
| Fictitious self-play, tabular | 2 | 5,000 | 0.2792 [-0.0914, 0.6498] | 0.1179 [-0.2331, 0.4689] | 0.26 |
| Naive self-play, DQN | 2 | 5,000 | 0.2542 [-0.3282, 0.8365] | 0.2489 [-0.4193, 0.9171] | 0.11 |
| NFSP (neural fictitious self-play) | 2 | 5,000 | 0.2917 [-0.2378, 0.8211] | 0.4993 [0.0080, 0.9907] | 0.13 |

Uniform random policy: 0.4583

Best any deterministic policy can do (1/6): 0.1667

CFR after 100 iterations: 0.00823; CFR+: 0.00119

## Welch's t-tests on final exploitability

| Comparison | Difference in means | t | p |
|---|---|---|---|
| Naive self-play, tabular Q (current) vs. Fictitious self-play, tabular (average) | +0.1613 | 1.92 | 0.27 |
| Naive self-play, tabular Q (average) vs. Fictitious self-play, tabular (average) | +0.0013 | 0.03 | 0.98 |
| Naive self-play, DQN (current) vs. NFSP (neural fictitious self-play) (average) | -0.2452 | -4.09 | 0.058 |
| Naive self-play, DQN (average) vs. NFSP (neural fictitious self-play) (average) | -0.2504 | -3.84 | 0.07 |
| Fictitious self-play, tabular (average) vs. NFSP (neural fictitious self-play) (average) | -0.3815 | -8.03 | 0.02 |
