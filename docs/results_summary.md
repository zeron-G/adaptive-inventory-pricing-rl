# Results Summary

## Experiment Run

The full experiment was run locally on April 28, 2026 using the public M5 Forecasting Accuracy dataset. The selected product-store series was:

- SKU: `FOODS_3_252`
- Store: `TX_1`
- Category: `FOODS`
- Mean daily sales: 29.53 units
- Nonzero-sales ratio: 0.996
- Median observed sell price: $1.48
- Simulator capacity: 140 units
- State count: 405
- Action count: 12
- Q-table size: 4,860 state-action values

The machine had an NVIDIA GeForce RTX 5090 available, but the final tabular implementation uses NumPy on CPU. For this small finite Q-table, CPU execution avoids unnecessary device-transfer overhead and is more reproducible in GitHub Actions.

## Hyperparameter Optimization

The tuning stage searched both Q-learning and SARSA over:

- learning rate `alpha`: 0.05, 0.10, 0.20
- discount factor `gamma`: 0.90, 0.95, 0.99
- epsilon decay: 0.992, 0.996, 0.999
- minimum epsilon: 0.03, 0.05
- seeds: 13, 29, 47

This produced 54 hyperparameter configurations per algorithm and 324 training runs in the tuning stage. Final models were retrained for 3,000 episodes and evaluated over 120 held-out test episodes.

## Main Test Results

| Policy | Mean return | Fill rate | Stockout units | Avg. inventory | Price changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q Learning | 791.72 | 0.882 | 398.80 | 44.60 | 74.90 |
| SARSA | 461.01 | 0.769 | 580.54 | 16.01 | 70.89 |
| Static regular + reorder point | 337.64 | 0.754 | 866.20 | 22.28 | 0.00 |
| Inventory markdown heuristic | 282.23 | 0.697 | 652.47 | 11.74 | 30.37 |
| Regular + average demand order-up-to | 152.27 | 0.998 | 8.83 | 103.00 | 0.00 |
| Random valid action | 36.90 | 0.924 | 278.98 | 78.02 | 79.68 |

## Interpretation

Q-learning achieved the highest average test return. It learned a more aggressive profit-oriented policy than the simple baselines, accepting more stockout risk than the average-demand order-up-to policy but avoiding the extreme overstocking cost of that conservative baseline. This is exactly the tradeoff the project was designed to study: the profit-maximizing policy is not necessarily the policy with the highest fill rate.

SARSA also outperformed the standard baselines on mean return, but it produced lower fill rate and lower inventory than Q-learning in this run. One plausible explanation is that SARSA's on-policy updates learned under continued exploratory behavior, which made its policy less willing to carry inventory in some states. This should be discussed as an empirical result rather than a universal property.

The average-demand order-up-to baseline nearly eliminated stockouts, but it held much more inventory and therefore earned lower reward. This baseline is operationally attractive when service level is the dominant objective, but it is less attractive under the current profit-based reward.

## Sensitivity Analysis

The sensitivity analysis evaluated frozen learned policies under changed simulator assumptions. Q-learning stayed strongest in most scenarios. The main exception was `lead_time_1`, where the static reorder-point baseline achieved higher return than Q-learning. This is reasonable: when replenishment is very fast, a simple reorder rule can perform extremely well because the cost of waiting is smaller.

Under `lead_time_3`, Q-learning remained positive while the reorder-point and markdown heuristics degraded sharply. This indicates that the learned policy was more robust when replenishment became slower and the delayed consequences of ordering mattered more.

## Key Limitation

M5 provides observed sales, prices, and calendar features, but it does not include true inventory, replenishment orders, or lost demand. The results should therefore be interpreted as a tabular RL experiment in a simulator calibrated from real retail data, not as a causal estimate of Walmart's actual optimal policy.
