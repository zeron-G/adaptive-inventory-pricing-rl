# Reproducibility Guide

## Environment

Use Python 3.10 or newer. The project was run with Python 3.11.

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data

The full experiment uses the public M5 Forecasting Accuracy data from Zenodo:

```powershell
python scripts/download_m5.py
```

Raw and processed data are intentionally excluded from git because the M5 files are large.

## Full Experiment

```powershell
python scripts/run_experiments.py --tuning-episodes 600 --final-episodes 3000 --evaluation-episodes 120
python scripts/build_report_notebook.py
pytest
```

Generated outputs:

- `reports/tables/hyperparameter_tuning_results.csv`
- `reports/tables/training_history.csv`
- `reports/tables/evaluation_summary.csv`
- `reports/figures/*.png`
- `reports/models/*.npy`
- `notebooks/Adaptive_Inventory_Pricing_Tabular_RL_Report.ipynb`

## Fast Smoke Run

CI uses a synthetic dataset so that tests do not depend on external downloads:

```powershell
python scripts/run_experiments.py --synthetic --tuning-episodes 4 --final-episodes 8 --evaluation-episodes 3 --reports-dir reports_ci
pytest
```

## Design Choices That Preserve Tabular RL

- State components are finite bins.
- Actions are finite combinations of price tier and order quantity.
- Q-learning and SARSA use an explicit Q-table.
- No neural-network function approximation is used as the primary method.
- M5 data calibrates demand; inventory transitions are simulated.
