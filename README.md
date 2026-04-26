# Adaptive Inventory & Pricing with Tabular RL

This repository contains the project planning and proposal materials for a BU.520.750.51.SP26 group project on adaptive retail inventory and pricing decisions using tabular reinforcement learning.

## Repository Contents

- `docs/project_plan_en.md`: full project plan in English.
- `docs/project_plan_zh.md`: independent Chinese translation of the full project plan.
- `docs/research_notes.md`: dataset scan, source links, and design notes.
- `proposal/group_project_proposal_detailed.md`: expanded proposal draft with detailed reasoning and implementation design.
- `proposal/group_project_proposal.md`: submission-ready one-page proposal draft in English.
- `proposal/group_project_proposal.zh.md`: Chinese reference translation of the proposal.
- `proposal/group_project_proposal.pdf`: one-page PDF generated from the proposal markdown.
- `scripts/build_proposal_pdf.py`: script used to generate the PDF.
- `data/README.md`: data download and usage notes.

## Important Placeholder

The current group metadata is:

- Group Number: Group Project 13
- Group Members: Jiayi Zhuo, Keyang Li, Rongze Gao, Zhexi Wang

## Proposed Core Method

The project will use a custom finite-state simulator calibrated from public retail sales data. The primary RL methods will be tabular Q-learning and SARSA, with optional value iteration on an estimated transition model. The state and action spaces are deliberately discretized to satisfy the course requirement that the project use tabular reinforcement learning.

## Data Direction

Primary dataset:

- M5 Forecasting - Accuracy, Walmart item-store daily sales with price and calendar features: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Backup / prototype datasets:

- Store Item Demand Forecasting Dataset: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Retail Sales Promotions and Demand Forecasting: https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting

## Academic Ethics Note

External sources and GenAI assistance should be acknowledged in the final Canvas submission, per the Carey Academic Ethics Policy. A compact acknowledgement is already included at the end of the proposal draft.
