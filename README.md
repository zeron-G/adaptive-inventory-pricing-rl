# Adaptive Inventory & Pricing with Tabular RL

This repository contains the project planning and proposal materials for a BU.520.750.51.SP26 group project on adaptive retail inventory and pricing decisions using tabular reinforcement learning.

## Repository Contents

- `docs/project_plan_en.md`: full project plan in English.
- `docs/project_plan_zh.md`: independent Chinese translation of the full project plan.
- `docs/research_notes.md`: dataset scan, source links, and design notes.
- `docs/literature_review.md`: literature notes with APA references.
- `docs/methodology.md`: technical methodology for the implemented simulator and RL methods.
- `docs/reproducibility.md`: commands for reproducing data processing, experiments, and tests.
- `docs/results_summary.md`: concise summary of the completed training, evaluation, and sensitivity results.
- `src/inventory_pricing_rl/`: reusable Python package for data processing, simulator, agents, baselines, experiments, and plots.
- `scripts/`: command-line entry points for downloading data, running experiments, and building the report notebook.
- `.github/workflows/ci.yml`: GitHub Actions CI with tests and a synthetic smoke experiment.
- `proposal/group_project_proposal_detailed.md`: expanded proposal draft with detailed reasoning and implementation design.
- `proposal/group_project_proposal.md`: submission-ready one-page proposal draft in English.
- `proposal/group_project_proposal.zh.md`: Chinese reference translation of the proposal.
- `proposal/group_project_proposal.pdf`: one-page PDF generated from the proposal markdown.
- `notebooks/Adaptive_Inventory_Pricing_Tabular_RL_Complete_Standalone_Report.ipynb`: primary standalone report notebook with all code and analysis inline.
- `notebooks/Adaptive_Inventory_Pricing_Tabular_RL_Report.ipynb`: shorter artifact-based report notebook.
- `reports/`: generated experiment tables, figures, and learned Q-tables.
- `data/README.md`: data download and usage notes.

## Important Placeholder

The current group metadata is:

- Group Number: Group Project 13
- Group Members: Jiayi Zhuo, Keyang Li, Rongze Gao, Zhexi Wang

## Proposed Core Method

The project will use a custom finite-state simulator calibrated from public retail sales data. The primary RL methods will be tabular Q-learning and SARSA, with optional value iteration on an estimated transition model. The state and action spaces are deliberately discretized to satisfy the course requirement that the project use tabular reinforcement learning.

## Reproduce the Full Experiment

```powershell
python -m pip install -e ".[dev]"
python scripts/download_m5.py
python scripts/run_experiments.py --tuning-episodes 600 --final-episodes 3000 --evaluation-episodes 120
python scripts/build_report_notebook.py
pytest
```

For a quick CI-style smoke run without downloading M5:

```powershell
python scripts/run_experiments.py --synthetic --tuning-episodes 4 --final-episodes 8 --evaluation-episodes 3 --reports-dir reports_ci
```

## Data Direction

Primary dataset:

- M5 Forecasting - Accuracy, Walmart item-store daily sales with price and calendar features: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Backup / prototype datasets:

- Store Item Demand Forecasting Dataset: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset
- Retail Sales Promotions and Demand Forecasting: https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting

## Academic Ethics Note

External sources and GenAI assistance should be acknowledged in the final Canvas submission, per the Carey Academic Ethics Policy. A compact acknowledgement is already included at the end of the proposal draft.
