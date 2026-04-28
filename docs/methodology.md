# Methodology

## Problem Framing

The retailer chooses a price tier and replenishment quantity each day. Demand is uncertain and depends on calendar context, recent demand, and the price tier. Today's choices affect tomorrow's inventory and therefore future stockout risk, holding cost, and pricing flexibility. This creates a sequential decision problem, not merely a one-step forecasting task.

## MDP

State:

```text
s_t = (inventory_bin, demand_signal_bin, price_tier, calendar_type, pipeline_bin)
```

Action:

```text
a_t = (price_choice, order_quantity)
```

Reward:

```text
r_t = revenue
      - procurement_cost
      - fixed_order_cost
      - holding_cost
      - stockout_penalty
      - excess_inventory_penalty
      - price_change_penalty
```

The reward balances profit, customer service, and inventory discipline. Revenue alone would reward over-discounting and over-ordering. Profit without stockout penalties could tolerate bad service. The combined reward is more operationally meaningful.

## Algorithms

Q-learning:

```text
Q(s_t, a_t) <- Q(s_t, a_t) + alpha [r_t + gamma max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
```

SARSA:

```text
Q(s_t, a_t) <- Q(s_t, a_t) + alpha [r_t + gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
```

Both algorithms are trained with epsilon-greedy exploration and evaluated greedily after training.

## Data Handling

M5 sales, prices, and calendar features are merged into one selected SKU-store daily time series. The pipeline creates lagged rolling demand features and chronological train/validation/test splits. The simulator samples demand from train-calibrated empirical distributions, with explicit price-response assumptions.

## Evaluation

The final policies are compared with:

- random valid action,
- static regular price plus reorder point,
- regular price plus average-demand order-up-to,
- inventory markdown heuristic.

Metrics include cumulative reward, fill rate, stockout units, average inventory, average order quantity, and number of price changes.
