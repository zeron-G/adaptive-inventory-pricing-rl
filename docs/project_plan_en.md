# Project Plan: Adaptive Inventory & Pricing with Tabular Reinforcement Learning

## 1. Executive Summary

This project studies a practical retail decision problem: each day, a store manager must decide both how much inventory to replenish and what price level to offer for a selected product or small product category. These two decisions interact. Higher prices may increase unit margin but reduce demand; lower prices may accelerate sales but create stockout risk; aggressive replenishment improves availability but raises holding cost. The goal is to learn an adaptive policy that improves long-run profit while maintaining reasonable service levels.

The project will stay within the course scope by using finite, discrete state and action spaces and tabular RL methods. The primary algorithms will be Q-learning and SARSA. If time permits, we will also estimate a transition model from the simulator and compare value iteration or policy iteration against the model-free methods.

## 2. Research Motivation

Inventory management and pricing are classic sequential decision problems. A one-time predictive model can forecast demand, but it does not directly answer what action should be taken today when that action changes tomorrow's state. Replenishment and pricing decisions create delayed consequences: a price discount today may reduce inventory and increase future stockout probability, while a large order today may reduce lost sales but increase holding cost if demand slows down. This feedback loop makes reinforcement learning a natural fit.

The literature supports this framing. Recent retail pricing work explores Q-learning for adaptive price decisions, and inventory management studies often formulate replenishment as an MDP with stock levels, lead times, ordering costs, holding costs, and stockout penalties. The M5 forecasting competition data also provides a credible retail foundation because it contains Walmart item-store daily unit sales, prices, and calendar features.

## 3. Recommended Scope

To keep the project tractable and clearly tabular, the first implementation should focus on a single store and 1 to 5 SKUs from a stable product family in the M5 data, such as frequently sold foods or household products. A small category extension can be added later only if the single-SKU environment is working well.

Recommended first version:

- One decision period equals one day.
- One selected store, one SKU for the core experiment.
- Inventory capacity capped at a small integer, such as 40 or 60 units.
- Replenishment lead time of 1 or 2 days.
- Three price tiers: discount, regular, premium.
- Four order levels: none, small, medium, large.
- State features binned into a finite tuple.

This scope produces a Q-table that is easy to inspect and explain in the final demonstration.

## 4. Dataset Strategy

### Primary Dataset: M5 Forecasting - Accuracy

The M5 data is the strongest primary choice because it is public, widely used, and based on Walmart retail sales. It includes:

- Daily sales by product and store.
- Product hierarchy such as category and department.
- Store and state identifiers.
- Weekly sell prices.
- Calendar features and special events.

The dataset does not include true on-hand inventory or procurement costs. We will therefore use it to calibrate demand and price-context behavior, then run a custom simulator where inventory and cost parameters are explicitly defined. This is acceptable because the project requirement allows a simulator or real-world data, and our environment will be grounded in real retail demand patterns.

### Backup Dataset 1: Store Item Demand Forecasting Dataset

This Kaggle dataset is smaller and synthetic, with daily store-item sales, prices, promotions, weekdays, and months. It is useful for a fast prototype because price and promotion variables are directly included and the file size is manageable.

### Backup Dataset 2: Retail Sales Promotions and Demand Forecasting

This synthetic Kaggle dataset includes daily sales, discounts, pricing, promotions, inventory levels, and time-based factors. It can support a quick demonstration if M5 download or preprocessing becomes too large for the course timeline.

## 5. MDP Formulation

### State Space

The state at day `t` will be a finite tuple:

`s_t = (inventory_bin, demand_signal_bin, price_tier, calendar_type, pipeline_bin)`

Candidate bins:

- `inventory_bin`: stockout, low, medium, high, excess.
- `demand_signal_bin`: low, normal, high, based on rolling 7-day or 14-day demand quantiles.
- `price_tier`: discount, regular, premium.
- `calendar_type`: weekday, weekend, event, SNAP/event-related day where available.
- `pipeline_bin`: no outstanding order, small order arriving, large order arriving.

This yields approximately `5 x 3 x 3 x 4 x 3 = 540` states. With 12 to 15 actions, the Q-table remains small enough for tabular learning and visualization.

### Action Space

Each action combines a price decision and an order decision:

`a_t = (price_choice, order_quantity)`

Candidate discrete choices:

- `price_choice`: discount, regular, premium.
- `order_quantity`: 0, small pack, medium pack, large pack.

For example, action `(discount, medium)` means set the product to a discount price tier today and place a medium replenishment order.

### Reward Function

The reward is daily operating profit with service-level penalties:

`reward = revenue - procurement_cost - fixed_order_cost - holding_cost - stockout_penalty - excess_inventory_penalty`

Where:

- `revenue = selling_price x fulfilled_demand`
- `procurement_cost = unit_cost x order_quantity`
- `holding_cost = h x ending_inventory`
- `stockout_penalty = p x lost_sales`

This reward encourages the agent to balance margin, availability, and inventory efficiency. It avoids rewarding revenue alone, which could cause over-ordering, and avoids rewarding low inventory alone, which could cause repeated stockouts.

## 6. Environment Design

The environment will be a custom tabular simulator calibrated from retail data:

1. Select a store-SKU series from M5 with enough nonzero demand and observed price variation.
2. Convert daily sales, price, weekday, event, and recent-demand history into discrete context bins.
3. Estimate empirical demand distributions by context. If a bin is sparse, back off to broader category-level or store-level empirical distributions.
4. In each simulated day:
   - Agent observes the discrete state.
   - Agent chooses price tier and order level.
   - Demand is sampled from the calibrated empirical distribution with a simple price-tier adjustment.
   - Sales are fulfilled up to available inventory.
   - Holding, stockout, procurement, and ordering costs are computed.
   - Pending replenishment arrives after the chosen lead time.
5. Episodes run for a fixed horizon, such as 90 or 180 days.

The simulator will be simple enough to explain in class but realistic enough to show the tradeoff between pricing and replenishment.

## 7. RL Methodology

Primary methods:

- Tabular Q-learning: off-policy learning with epsilon-greedy exploration.
- Tabular SARSA: on-policy comparison to test whether a more cautious policy emerges under exploration.

Optional methods if time permits:

- Value iteration using an estimated transition and reward model from the simulator.
- Policy iteration for comparison on the same finite MDP.
- Multi-armed bandit price-only ablation to connect to Week 2 and show why inventory state matters.

Training details:

- Discount factor `gamma`: 0.90 to 0.99.
- Learning rate `alpha`: decayed or fixed grid search.
- Exploration `epsilon`: start high and decay.
- Random seeds: at least 10 simulation seeds for stable evaluation.
- Hyperparameter selection: based on validation episodes, not test performance.

No deep RL or continuous function approximation will be used as the primary approach.

## 8. Baselines

The trained RL policy should be compared against clear non-RL baselines:

- Static pricing plus reorder point policy `(s, S)`.
- Static regular price plus order-up-to average demand target.
- Heuristic markdown policy: discount when inventory is high, premium when inventory is low.
- Random valid action policy.
- Optional historical-price replay policy using observed M5 price tiers.

The `(s, S)` baseline is especially important because it is a standard inventory policy and easy to explain.

## 9. Evaluation Metrics

Primary metric:

- Average cumulative profit per episode on held-out simulation periods.

Secondary metrics:

- Fill rate: fulfilled demand divided by total demand.
- Stockout days.
- Average ending inventory.
- Inventory turnover.
- Price stability or number of price changes.
- Sensitivity to holding cost, stockout penalty, and lead time.

The final report should include learning curves, policy heatmaps, and a table comparing RL against baselines.

## 10. Proposed Deliverables

Minimum viable final project:

- Clean data preprocessing notebook or script.
- Custom finite MDP simulator.
- Q-learning and SARSA implementations.
- Baseline policies.
- Evaluation notebook with plots and tables.
- Final presentation with policy interpretation.

Stronger final project:

- Value iteration or policy iteration comparison.
- Sensitivity analysis across lead times and cost settings.
- Small multi-SKU extension or category-level experiment.
- Interactive demo showing how the learned policy reacts to stock and demand states.

## 11. Timeline

Week 6:

- Finalize proposal.
- Confirm group roles.
- Download sample dataset and select candidate SKU/store pairs.

Week 7:

- Build preprocessing pipeline.
- Define discrete state/action spaces.
- Implement simulator and baseline policies.

Week 8:

- Implement Q-learning and SARSA.
- Run first experiments.
- Debug reward scaling and transition logic.

Week 9:

- Tune hyperparameters.
- Add evaluation metrics and plots.
- Compare against baselines.

Week 10:

- Prepare final report, code cleanup, and presentation.
- Add sensitivity analysis if time allows.

## 12. Risks and Mitigations

Risk: M5 does not include true inventory.

Mitigation: Use M5 for demand calibration and make inventory dynamics explicit in the simulator. Clearly state assumptions for unit cost, holding cost, stockout penalty, and lead time.

Risk: Price changes in historical data are not randomized, so causal price elasticity is uncertain.

Mitigation: Treat price response as a simulator scenario rather than a causal claim. Run sensitivity analysis under low, medium, and high price elasticity settings.

Risk: State space grows too large.

Mitigation: Start with one SKU, coarse bins, and only five state components. Expand only after the tabular agent works.

Risk: The reward function may produce unintended behavior.

Mitigation: Track fill rate, stockout days, and ending inventory in addition to profit. Tune penalties and report sensitivity.

## 13. Suggested Team Roles

- Data lead: download and preprocess M5 or backup datasets; select SKU/store subset.
- Environment lead: implement simulator, state discretization, transitions, and reward function.
- RL lead: implement Q-learning, SARSA, and optional value iteration.
- Evaluation/report lead: baselines, plots, final proposal/report writing, and presentation.

For a smaller group, combine data and environment roles, then combine RL and evaluation roles.

## 14. Source Links

- M5 Forecasting - Accuracy, Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- M5Dataset documentation, sktime: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.datasets.forecasting.m5_competition.M5Dataset.html
- Makridakis et al., M5 competition background: https://www.sciencedirect.com/science/article/pii/S0169207021001187
- Dynamic Retail Pricing via Q-Learning: https://arxiv.org/abs/2411.18261
- Hybrid algorithm based on reinforcement learning for smart inventory management: https://link.springer.com/article/10.1007/s10845-022-01982-5
- Inventory management MDP notes: https://adityam.github.io/stochastic-control/mdps/inventory-management-revisited.html
- Store Item Demand Forecasting Dataset: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Retail Sales Promotions and Demand Forecasting: https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting

## 15. Academic Ethics Acknowledgment

The final submission should acknowledge the public datasets, cited references, course materials, and GenAI assistance used to draft, structure, or refine the proposal and implementation plan.
