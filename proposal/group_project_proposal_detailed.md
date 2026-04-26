# Group Project Proposal: Adaptive Inventory & Pricing with Tabular Reinforcement Learning

Group Number: Group Project 13  
Group Members: Jiayi Zhuo, Keyang Li, Rongze Gao, Zhexi Wang  
Course: BU.520.750.51.SP26  
Working Title: Adaptive Inventory & Pricing with Tabular Reinforcement Learning for Retail SKU Decisions  
Prepared for: Project Proposal Submission

## Title

Adaptive Inventory & Pricing with Tabular Reinforcement Learning for Retail SKU Decisions

This project proposes a tabular reinforcement learning framework for a practical retail operations problem: how a store should jointly decide replenishment quantity and selling price for a product over time. The title intentionally highlights both operational levers, inventory and pricing, because the main insight of the project is that these decisions should not be optimized separately. A pricing decision changes demand and therefore affects the future inventory state; a replenishment decision changes the availability of inventory and therefore affects whether a future price can be profitably used. The project will use a finite Markov Decision Process and course-covered tabular methods, especially Q-learning and SARSA, to learn adaptive policies under stochastic demand.

## Problem & "Why RL?"

### Business Problem

Retailers repeatedly face the question: "Given current inventory, recent demand, seasonality, and price level, what price should we set today and how much should we reorder?" This decision is common in grocery, household goods, consumer packaged goods, and e-commerce retail. The decision is especially important for products whose demand varies by weekday, event, promotion, season, or local customer behavior.

Traditional inventory systems often use a fixed reorder rule, such as ordering when inventory falls below a reorder point. Traditional pricing systems often use static prices, periodic promotions, or simple markdown rules. These approaches are understandable and operationally convenient, but they can be suboptimal when demand is uncertain and when price affects demand. For example:

- If inventory is low and demand is high, the retailer may want to avoid discounting because a discount could accelerate a stockout.
- If inventory is high and demand is weak, the retailer may want to lower the price to convert excess inventory into sales.
- If a replenishment order is already in the pipeline, the retailer may tolerate a temporary discount differently than if no replenishment is arriving.
- If tomorrow is a high-demand event day, the retailer may prefer to preserve inventory today or place a larger replenishment order.

The project will model this as a daily decision problem for one retail SKU or a small SKU subset. At each day, the agent observes a discretized state, chooses a price tier and replenishment quantity, then receives a reward based on sales revenue minus inventory-related costs and penalties.

### Why This Is Not Just Forecasting

A standard supervised learning model could forecast future demand from historical sales, price, and calendar features. Forecasting is useful, but it does not directly solve the decision problem. The retailer does not only want to know "what will demand be?" The retailer wants to know "what should we do now, given that today's action changes tomorrow's situation?"

The distinction matters because price and replenishment create delayed consequences:

- A discount today may increase immediate sales, but it may also deplete inventory and reduce future sales opportunities.
- A high price today may protect inventory, but it may lose profitable demand if customers are price-sensitive.
- A large order today may reduce stockouts later, but it creates holding cost and possible excess inventory.
- A small order today may reduce holding cost, but it creates a risk of lost sales during future high-demand periods.

These consequences cannot be fully evaluated by a one-step prediction model. They require optimizing a policy over time. Reinforcement learning is appropriate because it is designed for sequential decision-making under uncertainty, where an agent learns by interacting with an environment and receiving delayed rewards.

### Why Tabular RL Is Appropriate for This Course

The course project requirement emphasizes tabular RL methods and finite, discrete state and action spaces. This project is designed around that requirement. The state variables will be discretized into a small number of bins, and the actions will be a finite set of price and replenishment choices. The main methods will be:

- Tabular Q-learning.
- Tabular SARSA.
- Optional value iteration or policy iteration if we estimate a finite transition model from the simulator.

We will not use deep reinforcement learning, neural network function approximation, or continuous action optimization as the primary method. Any statistical estimation used in the project will support environment calibration, not replace the tabular RL policy.

### Project Objective

The objective is to learn a policy that maximizes long-run expected operating profit while maintaining reasonable inventory service levels. More specifically, the agent should learn when to:

- discount inventory to stimulate demand,
- keep the regular price,
- use a premium price when inventory is scarce or demand is strong,
- place no replenishment order,
- place a small, medium, or large replenishment order.

The final policy should be interpretable. We should be able to inspect the Q-table or policy heatmaps and explain why the agent chooses different actions under different inventory and demand conditions.

## MDP Formulation

This section is the technical core of the project. We formulate the retail decision problem as a finite Markov Decision Process with state space, action space, transition dynamics, and reward function.

### Time Step and Episode

The decision period will be one day. This matches the granularity of the primary dataset, M5 Forecasting - Accuracy, which provides daily unit sales for Walmart product-store combinations.

An episode will represent a fixed selling horizon, such as 90 or 180 days. At the beginning of each episode, the environment initializes inventory, pipeline inventory, demand context, and calendar position. During each day:

1. The agent observes the current discrete state.
2. The agent chooses a price tier and order quantity.
3. Demand is sampled from a calibrated demand distribution.
4. Sales are fulfilled up to available inventory.
5. The environment computes profit and penalties.
6. Inventory and pipeline orders update.
7. The next day's state is observed.

### State Space

The proposed state at day `t` is:

```text
s_t = (inventory_bin_t, demand_signal_bin_t, price_tier_t, calendar_type_t, pipeline_bin_t)
```

Each component is finite and observable or directly computable from simulated operations.

#### 1. Inventory Bin

Inventory is one of the most important state variables because it determines whether demand can be fulfilled. We will discretize on-hand inventory into five bins:

| Inventory bin | Meaning | Example threshold if capacity = 60 |
| --- | --- | --- |
| `stockout` | no units available | 0 |
| `low` | available inventory is risky | 1-10 |
| `medium` | normal operating range | 11-30 |
| `high` | more than near-term expected demand | 31-45 |
| `excess` | close to capacity or overstocked | 46-60 |

Thresholds can be adjusted after selecting the SKU. The thresholds should be based on SKU-level average demand and inventory capacity rather than chosen arbitrarily.

#### 2. Demand Signal Bin

The agent should know whether recent demand is weak, normal, or strong. We will compute a rolling recent-demand signal, such as average sales over the previous 7 or 14 days, then discretize it by historical quantiles:

| Demand signal bin | Meaning |
| --- | --- |
| `low` | recent demand below the 33rd percentile |
| `normal` | recent demand between the 33rd and 67th percentiles |
| `high` | recent demand above the 67th percentile |

This feature allows the agent to react differently to the same inventory level under different demand conditions. For example, 15 units may be enough when demand is low but risky when demand is high.

#### 3. Current Price Tier

The current price tier helps preserve the Markov property because demand and future decisions may depend on the price currently in effect. We will use three price tiers:

| Price tier | Meaning |
| --- | --- |
| `discount` | below the SKU's normal price |
| `regular` | normal historical price level |
| `premium` | above the SKU's normal price |

For M5, price tiers can be created relative to the product-store historical median price. For example, discount may mean price below 95% of median, regular may mean 95%-105% of median, and premium may mean above 105% of median. If a selected SKU has little historical price variation, we will still define simulated price tiers around the median price.

#### 4. Calendar Type

Retail demand changes by day type. The M5 data includes calendar and event information, including weekdays, weekends, events, and SNAP-related indicators. We will discretize calendar context into four types:

| Calendar type | Meaning |
| --- | --- |
| `weekday` | regular weekday |
| `weekend` | Saturday or Sunday |
| `event` | holiday, cultural event, sporting event, or special date |
| `snap_or_event` | SNAP-related day or event-related high-demand day, where applicable |

If overlap occurs, we will use a priority rule such as event over weekend over weekday, or we will merge sparse categories to keep the state space small.

#### 5. Pipeline Inventory Bin

Replenishment often has a lead time. If an order has been placed but has not arrived, the agent should account for that incoming supply. We will discretize pipeline inventory into:

| Pipeline bin | Meaning |
| --- | --- |
| `none` | no outstanding replenishment |
| `small` | small or medium order arriving soon |
| `large` | large order arriving soon |

This state component is important because the best action may differ when inventory is low but a large order is arriving tomorrow versus when no order is coming.

### State Space Size

Using the proposed bins:

```text
5 inventory bins x 3 demand bins x 3 price tiers x 4 calendar types x 3 pipeline bins = 540 states
```

This is small enough for a tabular Q-table. With 12 actions, the Q-table has:

```text
540 states x 12 actions = 6,480 state-action values
```

This size is feasible for tabular Q-learning and SARSA, and it is also interpretable enough for policy visualization.

### Action Space

Each daily action combines a pricing decision and a replenishment decision:

```text
a_t = (price_choice_t, order_quantity_t)
```

#### Price Choices

The price decision is discrete:

| Price choice | Meaning |
| --- | --- |
| `discount` | lower price to stimulate demand |
| `regular` | maintain normal price |
| `premium` | increase margin or ration scarce inventory |

The actual numerical price will be generated from the selected SKU's baseline price. For example:

```text
discount price = 0.90 x median historical price
regular price  = 1.00 x median historical price
premium price  = 1.10 x median historical price
```

These multipliers can be adjusted in sensitivity analysis.

#### Order Quantity Choices

The replenishment decision is also discrete:

| Order choice | Example quantity if capacity = 60 | Interpretation |
| --- | ---: | --- |
| `none` | 0 | no replenishment |
| `small` | 10 | cover short-term demand |
| `medium` | 20 | restore normal stock |
| `large` | 35 | prepare for high demand or recover from low stock |

The environment will enforce capacity constraints. If an order would exceed maximum inventory capacity after arrival, the excess may be blocked or penalized, depending on the simulator design.

#### Total Actions

With 3 price choices and 4 order choices:

```text
3 x 4 = 12 actions
```

Examples:

- `(discount, none)`: stimulate demand without ordering more.
- `(regular, medium)`: maintain standard price while replenishing to a safer level.
- `(premium, large)`: protect scarce inventory today while preparing for future demand.
- `(discount, large)`: clear current inventory while preparing for continued demand.

### Transition Dynamics

The next state depends on demand, sales, replenishment arrival, and calendar progression. The simulator transition can be written conceptually as:

```text
inventory_available_t = inventory_t + arrivals_t
demand_t ~ DemandDistribution(context_t, price_choice_t)
fulfilled_demand_t = min(inventory_available_t, demand_t)
lost_sales_t = max(demand_t - inventory_available_t, 0)
ending_inventory_t = inventory_available_t - fulfilled_demand_t
pipeline_{t+1} = update_pipeline(pipeline_t, order_quantity_t, lead_time)
inventory_{t+1} = ending_inventory_t
calendar_{t+1} = next_calendar_day
demand_signal_{t+1} = updated_recent_demand_bin
```

The demand distribution will be calibrated from historical data by context. For sparse contexts, we will use a backoff strategy:

1. Product-store-context distribution if enough observations exist.
2. Product-store distribution by calendar type.
3. Product category or department distribution.
4. Overall selected SKU empirical distribution.

Because the M5 data is observational, we will be careful not to claim that it reveals true causal price elasticity. Instead, the simulator will include explicit price-response assumptions. For example, if regular-price expected demand is `mu`, discount demand may be sampled around `mu x 1.10` and premium demand around `mu x 0.90`, with sensitivity tested under low, medium, and high elasticity scenarios.

### Reward Function

The daily reward is operating profit with service-level and inventory penalties:

```text
r_t =
    sales_revenue_t
    - procurement_cost_t
    - fixed_order_cost_t
    - holding_cost_t
    - stockout_penalty_t
    - excess_inventory_penalty_t
    - price_change_penalty_t
```

Where:

```text
sales_revenue_t = selling_price_t x fulfilled_demand_t
procurement_cost_t = unit_cost x order_quantity_t
fixed_order_cost_t = K x 1{order_quantity_t > 0}
holding_cost_t = h x ending_inventory_t
stockout_penalty_t = p x lost_sales_t
excess_inventory_penalty_t = e x max(ending_inventory_t - target_high_inventory, 0)
price_change_penalty_t = c x 1{price_choice_t != previous_price_t}
```

The price-change penalty is optional. It can represent operational friction or customer irritation from changing prices too frequently. If we include it, we will keep it small so that it discourages unnecessary volatility without preventing useful adaptive pricing.

### Why This Reward Encourages Desired Behavior

The reward function is designed to avoid common unintended behaviors:

- If we reward revenue only, the agent may over-discount or over-order because it ignores cost.
- If we reward profit without stockout penalties, the agent may tolerate too many lost sales.
- If holding cost is too low, the agent may order excessively to avoid stockouts.
- If stockout penalty is too high, the agent may overstock.
- If price-change penalty is too high, the agent may behave like a static pricing policy.

By tracking profit together with stockout and inventory metrics, we can tune the reward parameters and interpret whether the learned policy is operationally reasonable.

### Markov Property Discussion

The true retail system may depend on many hidden factors, such as competitor prices, local weather, promotion calendars, and customer substitution behavior. Our MDP is an approximation. We include recent demand signal, calendar type, price tier, inventory, and pipeline inventory to capture the most relevant information available to the decision maker. The discretized state is not perfect, but it is sufficient for a course-level tabular RL project and keeps the state space finite.

## Environment and Data Strategy

### Primary Data Source: M5 Forecasting - Accuracy

The primary dataset will be Kaggle's M5 Forecasting - Accuracy dataset. It is based on Walmart retail unit sales and includes daily sales at the product-store level, weekly sell prices, and calendar information. Public documentation describes the dataset as containing 30,490 individual time series, corresponding to 3,049 products across 10 Walmart stores. The products belong to categories such as Foods, Household, and Hobbies, and the stores are located in California, Texas, and Wisconsin. The dataset also includes sell prices and calendar events.

This makes M5 a strong choice because our project needs:

- product-store daily sales to estimate demand distributions,
- price information to define price tiers and simulate price-sensitive demand,
- calendar and event information to model demand seasonality,
- product/store identifiers to select a manageable SKU-store subset.

Relevant source links:

- Kaggle competition page: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- Zenodo mirror of the Kaggle dataset: https://zenodo.org/records/12636070
- sktime M5Dataset documentation: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.datasets.forecasting.m5_competition.M5Dataset.html
- M5 competition background paper: https://www.sciencedirect.com/science/article/pii/S0169207021001187

### Why We Need a Simulator

M5 contains sales and price history, but it does not provide true on-hand inventory, replenishment orders, procurement cost, stockout observations, or lost demand. Therefore, we should not claim that we can directly train an inventory-control agent from complete real-world inventory transitions.

Instead, we will use M5 to calibrate demand and calendar context, then build a custom simulator for inventory and pricing decisions. This approach is appropriate because the project instructions allow either a simulator or real-world data. Our environment will combine both:

- real retail sales data for demand patterns,
- simulated inventory dynamics for decisions and consequences.

This is also methodologically safer. We can clearly state which parts are observed from public data and which parts are modeling assumptions.

### Data Preprocessing Plan

The preprocessing pipeline will likely include these steps:

1. Load M5 sales, sell price, and calendar files.
2. Convert wide daily sales columns into a long format with one row per product-store-day.
3. Merge daily sales with calendar information by day ID.
4. Merge weekly sell price information by item, store, and week.
5. Select a candidate store-SKU pair or small SKU set using criteria such as:
   - enough nonzero sales days,
   - stable but nontrivial demand,
   - observed price variation if possible,
   - not too many long zero-sales periods.
6. Compute recent demand features, such as 7-day rolling mean and 14-day rolling mean.
7. Define discrete bins for demand signal, price tier, calendar type, inventory, and pipeline inventory.
8. Estimate empirical demand distributions by context.

### Candidate SKU Selection Criteria

The selected SKU should not be too sparse because tabular RL needs enough demand signal for learning. We will rank candidate SKU-store series using:

- average daily unit sales,
- percentage of nonzero sales days,
- variance or coefficient of variation,
- price variation,
- event/weekend demand differences.

A good candidate might be a frequently sold food or household product. A poor candidate would be a product with almost all zero-sales days, because the agent would learn little beyond "do not order."

### Simulator Design

The simulator will represent a periodic-review inventory system with price-dependent stochastic demand. The sequence for each day will be:

1. Receive any replenishment scheduled to arrive today.
2. Observe state.
3. Choose price tier and order quantity.
4. Sample demand using calendar context, recent demand bin, and price tier.
5. Fulfill demand up to available inventory.
6. Record lost sales if demand exceeds inventory.
7. Compute reward.
8. Add new order to the pipeline.
9. Advance to the next day.

Key simulator parameters:

| Parameter | Meaning | Initial value idea |
| --- | --- | --- |
| `capacity` | maximum on-hand inventory | 40-80 units depending on SKU |
| `lead_time` | days between order and arrival | 1 or 2 days |
| `unit_cost` | procurement cost per unit | 50%-70% of regular price |
| `fixed_order_cost` | fixed cost per nonzero order | small positive value |
| `holding_cost` | cost per unit left in inventory | 1%-3% of unit cost per day |
| `stockout_penalty` | penalty per unit of lost demand | lost margin or larger |
| `price_elasticity_scenario` | demand response to price | low/medium/high |

The exact values will be chosen for interpretability and tested with sensitivity analysis.

### Demand Modeling in the Simulator

The simplest demand model will be empirical sampling:

```text
base_demand_t ~ empirical distribution for selected context
adjusted_demand_t = price_adjustment(price_choice_t) x base_demand_t + noise
```

Example price adjustments:

| Scenario | Discount effect | Premium effect |
| --- | ---: | ---: |
| Low elasticity | +5% demand | -5% demand |
| Medium elasticity | +15% demand | -15% demand |
| High elasticity | +30% demand | -30% demand |

Demand will be rounded to nonnegative integer units. We can also cap extreme sampled demand to avoid unrealistic outliers in a small course project.

### Backup Data Sources

If M5 preprocessing becomes too time-consuming, we will use smaller Kaggle datasets for prototyping:

- Store Item Demand Forecasting Dataset: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Retail Sales Promotions and Demand Forecasting: https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting
- Corporacion Favorita Grocery Sales Forecasting: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

These backup datasets are useful for faster experiments, but M5 remains the preferred source because it is a well-known retail dataset with daily sales, prices, and calendar information.

### RL Training Methodology

#### Q-learning

Q-learning is an off-policy tabular method. It updates the state-action value table using:

```text
Q(s_t, a_t) <- Q(s_t, a_t) + alpha[
    r_t + gamma max_a Q(s_{t+1}, a) - Q(s_t, a_t)
]
```

This method learns the value of the greedy target policy while exploring with an epsilon-greedy behavior policy. It is appropriate for our project because the state-action table is small and finite.

#### SARSA

SARSA is an on-policy tabular method. It updates using the action actually selected in the next state:

```text
Q(s_t, a_t) <- Q(s_t, a_t) + alpha[
    r_t + gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
]
```

SARSA may learn a more conservative policy under exploration because it accounts for the behavior policy's future exploratory actions. Comparing Q-learning and SARSA is useful because inventory systems can be sensitive to risky exploratory decisions.

#### Optional Value Iteration or Policy Iteration

If time permits, we will estimate a finite transition model from simulator rollouts:

```text
P(s' | s, a), E[r | s, a]
```

Then we can run value iteration or policy iteration. This would connect directly to Week 4 and provide a model-based comparison. However, Q-learning and SARSA will remain the primary methods because they are simpler to implement and align well with simulator interaction.

#### Exploration Strategy

We will use epsilon-greedy exploration:

```text
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay over training episodes
```

Early training will explore many price/order combinations. Later training will exploit learned high-value actions. We may restrict obviously infeasible actions, such as placing a large order when inventory and pipeline already exceed capacity.

#### Hyperparameters

Candidate hyperparameters:

| Hyperparameter | Candidate values |
| --- | --- |
| `gamma` | 0.90, 0.95, 0.99 |
| `alpha` | 0.05, 0.10, 0.20 |
| `epsilon_decay` | linear decay, exponential decay |
| episode length | 90 days, 180 days |
| number of episodes | 5,000-50,000 depending on runtime |

Hyperparameters will be selected using validation episodes. Final evaluation will be run on held-out random seeds or held-out calendar periods.

## Baselines and Evaluation

### Baseline Strategies

We will compare the RL policies against multiple baselines so that improvement is meaningful and interpretable.

#### 1. Static Regular Price + Reorder Point Policy

This baseline uses a fixed regular price. It places an order when inventory position falls below a reorder point.

```text
if on_hand_inventory + pipeline_inventory <= reorder_point:
    order up to target_level
else:
    order 0
```

This is a standard and practical inventory heuristic. It is likely to perform reasonably well, making it a useful benchmark.

#### 2. Static Regular Price + Average-Demand Order-Up-To Policy

This baseline orders enough inventory to cover expected demand over a fixed number of future days. It is simple and forecast-like:

```text
target_inventory = average_daily_demand x coverage_days
order_quantity = max(target_inventory - inventory_position, 0)
```

It helps answer whether RL improves over a simple demand-based planning rule.

#### 3. Heuristic Markdown Policy

This baseline changes price based on inventory:

```text
if inventory_bin == excess:
    price = discount
elif inventory_bin == stockout or low:
    price = premium
else:
    price = regular
```

The order quantity can follow the reorder point rule. This baseline is important because it directly captures a common human intuition: discount when inventory is high and raise price when inventory is scarce.

#### 4. Random Valid Action Policy

This policy randomly selects among feasible actions. It is not a practical business policy, but it confirms that trained RL is learning something better than random behavior.

#### 5. Optional Historical Price Replay Policy

If suitable price data is available for the selected SKU, we can replay historical price tiers from M5 and combine them with a fixed replenishment rule. This baseline gives a rough comparison to historical pricing patterns, although it should not be treated as the true Walmart policy because inventory decisions are not observed.

### Evaluation Design

The agent will be trained in simulated episodes and evaluated separately. The evaluation should avoid measuring performance on the same randomness used for training.

Recommended design:

1. Split calendar periods or random seeds into training, validation, and testing.
2. Train Q-learning and SARSA over many episodes.
3. Select hyperparameters using validation performance.
4. Freeze learned policy.
5. Evaluate frozen policy over test episodes and multiple random seeds.
6. Compare all methods using the same test conditions.

### Primary Metric

The main metric will be average cumulative profit per episode:

```text
Average Test Profit = mean over test episodes of sum_t r_t
```

This directly aligns with the business objective, but it is not sufficient by itself because a high-profit policy could still create poor service levels or unstable prices.

### Secondary Metrics

We will report several operational metrics:

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Fill rate | fulfilled demand / total demand | measures customer service |
| Stockout days | days with zero inventory or lost sales | identifies service failures |
| Average ending inventory | average units left after demand | measures overstock risk |
| Inventory turnover | sales / average inventory | measures inventory efficiency |
| Gross margin | revenue - procurement cost | separates margin from penalties |
| Price changes | number of price-tier changes | measures operational stability |
| Average selling price | mean realized price | helps interpret pricing behavior |

### Visual Evaluation

We will include:

- learning curves showing episode reward over training,
- profit comparison bar charts,
- fill rate and stockout comparison charts,
- heatmaps of learned policy by inventory bin and demand signal,
- Q-value tables for selected states,
- sensitivity plots for holding cost, stockout penalty, lead time, and price elasticity.

The policy heatmaps are especially useful for demonstrating that the tabular agent learned interpretable behavior. For example, we expect the learned policy to discount more often when inventory is high and demand is low, and to order more aggressively when demand is high and pipeline inventory is empty.

### Expected Results

We expect Q-learning and SARSA to outperform the random policy and simple static baselines in cumulative profit. The comparison against the heuristic markdown policy may be more competitive. RL should perform better if it learns interactions that the heuristic does not capture, such as:

- ordering ahead of event days,
- avoiding discounts when pipeline inventory is empty,
- using premium pricing during low inventory states,
- avoiding large orders when recent demand is weak.

Q-learning may achieve higher profit but slightly riskier inventory behavior. SARSA may produce a more conservative policy because it learns under the exploration policy. This comparison will be interesting and aligned with course concepts.

### Sensitivity Analysis

Because some simulator assumptions are not directly observed from M5, we will run sensitivity analysis. We will test:

- low, medium, and high price elasticity,
- low and high holding cost,
- low and high stockout penalty,
- 1-day and 2-day lead time,
- different inventory capacities,
- different selected SKUs.

This will show whether the learned policy is robust or dependent on a narrow parameter setting.

### Limitations

The project has several limitations that we will state clearly:

1. M5 reports sales, not true demand. If a product was out of stock historically, observed sales may understate demand.
2. M5 does not include on-hand inventory or real replenishment orders, so inventory must be simulated.
3. Historical price variation is observational, not randomized, so price elasticity cannot be causally identified from M5 alone.
4. The finite state representation simplifies real retail operations.
5. Single-SKU optimization ignores substitution and basket effects.

These limitations do not invalidate the project because the goal is to demonstrate tabular RL on a well-defined finite simulator calibrated by real retail data, not to deploy a production retail pricing engine.

### Team Roles

The team has four members, so the work can be divided as follows:

| Member | Suggested role | Responsibilities |
| --- | --- | --- |
| Jiayi Zhuo | Data and preprocessing lead | download data, clean M5 files, select SKU/store, build discrete context bins |
| Keyang Li | Environment and simulator lead | implement inventory dynamics, demand sampling, reward function, state/action encoding |
| Rongze Gao | RL methodology lead | implement Q-learning, SARSA, optional value iteration/policy iteration, tune hyperparameters |
| Zhexi Wang | Evaluation and reporting lead | implement baselines, generate plots/tables, write final interpretation and presentation materials |

These roles can be adjusted, but assigning ownership early will reduce duplicated work.

### Proposed Timeline

| Week | Work plan |
| --- | --- |
| Week 6 | Finalize proposal, confirm data source, define MDP, assign roles |
| Week 7 | Download/process M5 sample, choose SKU/store, implement simulator skeleton |
| Week 8 | Implement Q-learning and SARSA, run initial experiments, debug reward and transitions |
| Week 9 | Add baselines, tune hyperparameters, run evaluation across seeds and scenarios |
| Week 10 | Prepare final report, plots, policy interpretation, and presentation/demo |

### Final Deliverables

Expected final project deliverables:

- source code for preprocessing,
- finite-state inventory-pricing simulator,
- Q-learning and SARSA implementation,
- baseline policies,
- evaluation notebook or scripts,
- plots and tables comparing performance,
- final report and presentation,
- clear statement of assumptions and limitations.

### References and External Resources

- Kaggle. M5 Forecasting - Accuracy dataset. https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- Zenodo mirror. M5 Forecasting Accuracy dataset. https://zenodo.org/records/12636070
- sktime documentation. M5Dataset. https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.datasets.forecasting.m5_competition.M5Dataset.html
- Makridakis et al. The M5 competition: Background, organization, and implementation. International Journal of Forecasting. https://www.sciencedirect.com/science/article/pii/S0169207021001187
- Rana et al. Dynamic Retail Pricing via Q-Learning: A Reinforcement Learning Framework for Enhanced Revenue Management. https://arxiv.org/abs/2411.18261
- NEASQC. Reinforcement learning for inventory management. https://www.neasqc.eu/use-case/reinforcement-learning-for-inventory-management/
- Aditya Mahajan. Inventory management revisited. https://adityam.github.io/stochastic-control/mdps/inventory-management-revisited.html
- Store Item Demand Forecasting Dataset. https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Retail Sales Promotions and Demand Forecasting. https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting

### Academic Ethics Acknowledgment

This proposal was prepared using course concepts, public dataset documentation, published and online references listed above, and GenAI assistance from ChatGPT/Codex for research organization, drafting, and editing. All final submitted work will be reviewed and revised by the group members.
