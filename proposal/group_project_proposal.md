Group Number: Group Project 13  
Group Members: Jiayi Zhuo, Keyang Li, Rongze Gao, Zhexi Wang  
Course: BU.520.750.51.SP26  
Date: April 26, 2026

## Title

Adaptive Inventory & Pricing with Tabular Reinforcement Learning for Retail SKU Decisions

## Problem & "Why RL?"

Retail managers repeatedly decide how much stock to reorder and what price to set. A one-time prediction model can forecast demand, but it cannot directly optimize actions whose effects carry into future periods. A discount may increase today's sales but raise tomorrow's stockout risk; a large replenishment order may protect service levels but create holding costs if demand falls. This makes the problem a sequential decision under uncertainty, where the agent must learn a policy that balances margin, availability, and inventory cost over time.

## MDP Formulation

State Space: We will use a finite tuple observed each day: `(inventory_bin, demand_signal_bin, price_tier, calendar_type, pipeline_bin)`. Inventory is discretized into stockout/low/medium/high/excess; demand signal into low/normal/high using recent sales quantiles; price into discount/regular/premium; calendar into weekday/weekend/event/SNAP-related day where available; and pipeline into no/small/large outstanding replenishment.

Action Space: Each action is discrete: `(price_choice, order_quantity)`, where price choice is discount/regular/premium and order quantity is none/small/medium/large.

Reward Function: Daily reward equals operating profit: sales revenue minus procurement, fixed ordering, holding, stockout, and excess-inventory penalties. This rewards profitable sales while discouraging both chronic stockouts and unnecessary overstocking.

Methodology: The primary methods will be tabular Q-learning and SARSA, with optional value iteration or policy iteration on an estimated transition model. Deep RL and continuous function approximation will not be used as primary methods.

## Environment and Data Strategy

We will build a custom finite-state simulator calibrated from public retail data. The primary dataset is Kaggle's M5 Forecasting - Accuracy Walmart dataset, which includes item-store daily sales, sell prices, and calendar events. Since true inventory is not reported, inventory dynamics, lead time, unit cost, holding cost, and stockout penalty will be simulated explicitly. We will start with one store-SKU pair or a small SKU set, discretize all variables into finite bins, and sample demand from empirical context distributions. Smaller Kaggle retail datasets with price, promotion, and inventory fields will be retained as backups for prototyping.

## Baselines and Evaluation

We will compare the RL agent against static pricing with an `(s, S)` reorder policy, regular-price order-up-to-average-demand, a simple markdown heuristic, and a random valid-action policy. Evaluation will use held-out simulated episodes and multiple random seeds. Primary performance will be average cumulative profit; secondary metrics will include fill rate, stockout days, average ending inventory, inventory turnover, and number of price changes. Learning curves and policy heatmaps will show whether the tabular agent learns interpretable behavior.

Acknowledgment: We used course materials, public dataset documentation, research articles on Q-learning for pricing/inventory, and GenAI assistance from ChatGPT/Codex to draft and organize this proposal.
