# Research Notes

## Short Recommendation

Use the M5 Forecasting - Accuracy dataset as the primary data source and build a custom finite-state simulator around one store-SKU pair. This gives the project real retail data credibility while keeping the RL method fully tabular.

## Dataset Comparison

| Dataset | Link | Strengths | Limitations | Recommended Use |
| --- | --- | --- | --- | --- |
| M5 Forecasting - Accuracy | https://www.kaggle.com/competitions/m5-forecasting-accuracy/data | Real Walmart retail sales, daily item-store granularity, sell prices, calendar events, hierarchy | Large files, no true on-hand inventory, price causality is not guaranteed | Primary dataset for demand calibration |
| Store Item Demand Forecasting Dataset | https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset | Smaller, daily store-item sales, prices, promotions, weekday/month fields, MIT license per Kaggle page | Synthetic rather than real retail | Prototype or backup |
| Retail Sales Promotions and Demand Forecasting | https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting | Small, includes inventory levels, price, discount, promotions, public domain per Kaggle page | Synthetic and only 2,800 records according to Kaggle page | Backup demo if M5 is too heavy |
| Corporacion Favorita Grocery Sales Forecasting | https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting | Large grocery sales data, promotions, holidays, stores, oil prices | Large, older competition, no direct inventory | Alternative if team prefers grocery domain |

## Why M5 Fits the Proposal

The M5 competition paper describes a Walmart retail sales forecasting task with 42,840 hierarchical unit sales series and exogenous variables. The sktime M5Dataset documentation lists the three main files as daily sales, sell prices, and calendar information including events. These are exactly the ingredients needed to create a realistic simulated demand environment for price and inventory decisions.

The main gap is inventory. Because M5 does not report on-hand stock, the project should not claim to reconstruct real Walmart inventory. Instead, it should explicitly simulate inventory and use M5 only for demand/context calibration.

## Methodology Guardrails for the Course

The Canvas discussion emphasizes finite, discrete state and action spaces and tabular RL methods. The project should therefore:

- Use Q-learning and SARSA as primary methods.
- Keep state bins coarse and explainable.
- Keep actions discrete: a few price tiers and order quantities.
- Avoid deep RL, neural networks, and continuous function approximation as the main policy approach.
- Treat any regression or smoothing as environment calibration only, not as the RL method.

## Suggested State and Action Cardinality

State tuple:

`(inventory_bin, demand_signal_bin, price_tier, calendar_type, pipeline_bin)`

Cardinality:

- 5 inventory bins.
- 3 demand signal bins.
- 3 price tiers.
- 4 calendar types.
- 3 pipeline bins.

Total states: 540.

Action tuple:

`(price_choice, order_quantity)`

Cardinality:

- 3 price choices.
- 4 order quantities.

Total actions: 12.

Q-table size: 6,480 state-action values, which is comfortably tabular.

## Literature Links Used

- M5 competition background, organization, and implementation: https://www.sciencedirect.com/science/article/pii/S0169207021001187
- M5Dataset documentation: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.datasets.forecasting.m5_competition.M5Dataset.html
- Dynamic retail pricing with Q-learning: https://arxiv.org/abs/2411.18261
- Q-learning for smart inventory management: https://link.springer.com/article/10.1007/s10845-022-01982-5
- Inventory management MDP teaching notes: https://adityam.github.io/stochastic-control/mdps/inventory-management-revisited.html

## Final Proposal Positioning

The proposal should not overpromise causal price optimization. A careful wording is:

"We will use public sales and price data to calibrate a finite simulator, then evaluate tabular RL policies under explicit price-response and cost assumptions."

This is academically safer than saying:

"We will learn the true optimal Walmart pricing policy from historical data."
