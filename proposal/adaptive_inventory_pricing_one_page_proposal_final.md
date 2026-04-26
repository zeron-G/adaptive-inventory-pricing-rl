# Adaptive Inventory & Pricing with Tabular Reinforcement Learning

**Group Project 13** | **Jiayi Zhuo, Keyang Li, Rongze Gao, Zhexi Wang** | **BU.520.750.51.SP26**

## Project Focus and Business Motivation

This project studies a practical retail operations problem: each day, a store must jointly choose a selling price and replenishment quantity for a SKU under uncertain demand. Treating pricing and inventory separately is often suboptimal because price changes demand and future inventory, while replenishment determines whether future demand can be served profitably.

The business objective is to maximize long-run operating profit while maintaining service quality. The learned policy should know when to discount excess inventory, hold regular price, use premium price when inventory is scarce, and place no/small/medium/large replenishment orders depending on demand context, pipeline inventory, and calendar conditions.

## Why Reinforcement Learning

A demand forecast answers “what may happen,” but the decision problem asks “what should we do now.” A discount can increase immediate sales but create tomorrow’s stockout; a large order can reduce lost sales but create holding cost and overstock. These delayed consequences make the problem sequential, so reinforcement learning is appropriate: actions are evaluated by immediate reward plus future value, not only one-step prediction accuracy.

The project remains aligned with the course by using finite, discrete state and action spaces and tabular methods rather than deep RL or continuous control. Any statistical model will support demand calibration, not replace the RL policy.

## MDP Formulation

**Time step:** one day.  
**Episode:** a 90- or 180-day selling horizon.  
At each day, the agent observes a discretized state, chooses a price tier and order quantity, demand is sampled, sales are fulfilled subject to inventory, reward is computed, and inventory/pipeline/calendar states update.

**State:** inventory bin, recent-demand signal, current price tier, calendar type, and pipeline bin. Inventory uses stockout, low, medium, high, and excess. Recent demand uses low, normal, and high based on rolling 7/14-day demand quantiles. Price has discount, regular, and premium. Calendar type includes weekday, weekend, event, and SNAP/event-like days. Pipeline inventory is none, small, or large incoming order.

**Scale:** 5 inventory bins × 3 demand bins × 3 price tiers × 4 calendar types × 3 pipeline bins = **540 states**. With **12 actions**, the Q-table has **6,480 state-action values**, which is small enough for tabular learning and policy heatmap interpretation.

**Actions:** each action combines a price choice and an order quantity. Price choices are discount/regular/premium; order choices are none/small/medium/large. Numerical prices can be 0.90×, 1.00×, and 1.10× of the selected SKU median historical price; order sizes are calibrated to capacity and average demand.

**Transitions:** inventory receives scheduled arrivals; stochastic demand is sampled from M5-calibrated context distributions and adjusted by explicit price-response assumptions; sales equal min(inventory, demand); lost sales are recorded; new orders enter the pipeline with 1–2 day lead time; and the next calendar and rolling-demand state are updated.

**Reward:** daily operating profit = sales revenue − procurement cost − fixed order cost − holding cost − stockout penalty − excess-inventory penalty − optional price-change penalty. This discourages revenue-only behavior, excessive ordering, frequent price oscillation, and policies that accept too many lost sales.

## Data and Simulator Strategy

Primary data will be the **M5 Forecasting - Accuracy** retail dataset, which provides Walmart daily unit sales, weekly sell prices, and calendar/event information for product-store time series. We will select one SKU-store pair or a small SKU subset with sufficient nonzero sales, meaningful variation, and usable price/calendar context.

M5 gives realistic demand seasonality and price history, but it does not include true on-hand inventory, replenishment orders, or lost demand. Therefore, we will build a custom simulator calibrated by M5 instead of claiming direct real-world inventory transitions. This is methodologically safer and matches the project allowance for simulator or real-world data.

Preprocessing will convert sales from wide to long format, merge calendar and price files, construct rolling 7/14-day demand signals, define price tiers relative to median price, classify calendar days, and estimate empirical demand distributions by context. Sparse contexts will back off from product-store-context to SKU/calendar, category, or overall empirical distributions.

Candidate SKU selection will consider average daily sales, share of nonzero sales days, coefficient of variation, price variation, and event/weekend demand differences. A good candidate is frequent enough to learn from but volatile enough to make inventory/pricing decisions meaningful.

Demand will be sampled from calibrated base demand and adjusted by elasticity scenarios. For example, low/medium/high elasticity may set discount demand to +5%/+15%/+30% and premium demand to −5%/−15%/−30%. Sensitivity analysis will test whether conclusions depend on these assumptions.

Key simulator parameters include capacity, lead time, unit procurement cost, fixed order cost, holding cost, stockout penalty, and excess-inventory penalty. Initial values will be chosen for interpretability and then varied to test robustness.

## RL Methods and Baselines

Main algorithms: **tabular Q-learning** and **tabular SARSA** with epsilon-greedy exploration and decaying epsilon. Q-learning may learn a higher-profit but more aggressive greedy policy; SARSA may be more conservative because it learns under the exploratory behavior policy. If time permits, we may estimate transition probabilities and expected rewards from simulator rollouts and run value iteration or policy iteration as a model-based comparison.

Candidate hyperparameters include discount factor 0.90/0.95/0.99, learning rate 0.05/0.10/0.20, linear or exponential epsilon decay, episode length 90 or 180 days, and 5,000–50,000 training episodes depending on runtime. Hyperparameters will be selected using validation episodes.

Baselines: (1) static regular price plus reorder-point policy; (2) regular price plus average-demand order-up-to rule; (3) heuristic markdown policy that discounts excess inventory and premiums scarce inventory; (4) random valid action policy; and optionally (5) historical price replay with a fixed replenishment rule if the SKU has enough price variation.

## Evaluation Plan

We will train on simulated episodes and evaluate frozen policies on held-out random seeds or calendar periods. The primary metric is average cumulative test profit. Secondary operational metrics include fill rate, stockout days, lost sales, average ending inventory, inventory turnover, gross margin, number of price changes, and average selling price.

Visual outputs will include learning curves, profit/fill-rate comparison charts, policy heatmaps by inventory and demand state, selected Q-value tables, and sensitivity plots for holding cost, stockout penalty, lead time, capacity, SKU choice, and price elasticity.

## Implementation Timeline

**Week 1:** finalize SKU selection, preprocessing, and demand-context bins.  
**Week 2:** build and unit-test the finite-state simulator, reward function, and baseline policies.  
**Week 3:** train Q-learning and SARSA, tune alpha, gamma, and epsilon decay, and compare validation performance.  
**Week 4:** run test evaluation, sensitivity analysis, policy heatmaps, and prepare the final report and presentation.

## Expected Insights, Limitations, and Deliverables

We expect RL to outperform random and simple static rules, while the heuristic markdown baseline may be competitive. The key contribution is interpretable policy behavior: discount high inventory with weak demand, avoid discounts when inventory is low and no pipeline order is coming, order ahead of high-demand/event days, and reduce ordering when recent demand is weak.

Limitations will be stated clearly: M5 observes sales rather than true demand; inventory and replenishment are simulated; price elasticity is assumed rather than causally identified; single-SKU control ignores substitution and basket effects; and discretization simplifies real retail operations.

Final deliverables include preprocessing code, finite-state simulator, Q-learning/SARSA implementation, baseline policies, evaluation notebook, plots/tables, final report, presentation, and a clear assumptions/ethics statement.

## References and Academic Ethics Acknowledgment

Key resources: Kaggle M5 Forecasting - Accuracy dataset; Zenodo M5 mirror; sktime M5Dataset documentation; Makridakis et al., *The M5 competition*; Rana et al., *Dynamic Retail Pricing via Q-Learning*; NEASQC inventory-management RL case; Aditya Mahajan inventory-management MDP notes.

This proposal uses course concepts, public dataset documentation, published/online references, and GenAI assistance from ChatGPT/Codex for drafting, organization, and editing. The group will review and revise all final submitted work.
