"""Evaluate learned policies under perturbed simulator assumptions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inventory_pricing_rl.agents import evaluate_policy, greedy_policy
from inventory_pricing_rl.baselines import BASELINE_POLICIES
from inventory_pricing_rl.data import load_processed_series
from inventory_pricing_rl.environment import EnvironmentConfig
from inventory_pricing_rl.experiments import make_env_factory


def main() -> None:
    processed_dir = Path("data/processed/m5_subset")
    reports_dir = Path("reports")
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    models_dir = reports_dir / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    daily_data, meta = load_processed_series(processed_dir)
    base_config = EnvironmentConfig.from_metadata(meta)
    base_params = asdict(base_config)

    scenarios = [
        ("base", "base", {}),
        ("elasticity_low", "price_elasticity", {"discount_demand_lift": 0.05, "premium_demand_drop": 0.05}),
        ("elasticity_high", "price_elasticity", {"discount_demand_lift": 0.30, "premium_demand_drop": 0.30}),
        ("lead_time_1", "lead_time", {"lead_time": 1}),
        ("lead_time_3", "lead_time", {"lead_time": 3}),
        ("holding_low", "holding_cost", {"holding_cost_rate": 0.005}),
        ("holding_high", "holding_cost", {"holding_cost_rate": 0.030}),
        ("stockout_low", "stockout_penalty", {"stockout_penalty_rate": 0.40}),
        ("stockout_high", "stockout_penalty", {"stockout_penalty_rate": 1.00}),
    ]

    learned = {}
    for algorithm in ["q_learning", "sarsa"]:
        q_path = models_dir / f"{algorithm}_q_table.npy"
        if q_path.exists():
            learned[algorithm.replace("_", " ").title()] = greedy_policy(np.load(q_path))

    rows = []
    for scenario_name, scenario_type, overrides in scenarios:
        params = base_params.copy()
        params.update(overrides)
        env_config = EnvironmentConfig(**params)
        env_factory = make_env_factory(daily_data, env_config)
        policies = {**learned, **BASELINE_POLICIES}
        for policy_name, policy in policies.items():
            eval_df = evaluate_policy(
                env_factory,
                policy,
                split="test",
                episodes=60,
                seed=20260428,
                label=policy_name,
            )
            grouped = eval_df[["return", "fill_rate", "stockout_units", "avg_inventory"]].mean()
            rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_type": scenario_type,
                    "policy": policy_name,
                    **grouped.to_dict(),
                }
            )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(tables_dir / "sensitivity_analysis.csv", index=False)

    pivot = sensitivity.pivot_table(index="scenario", columns="policy", values="return", aggfunc="mean")
    pivot = pivot.loc[[name for name, _, _ in scenarios]]
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Sensitivity analysis: mean test return under perturbed assumptions")
    ax.set_ylabel("Mean cumulative reward")
    ax.set_xlabel("Scenario")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_returns.png", dpi=170)
    plt.close(fig)

    print("Sensitivity analysis complete.")
    print(sensitivity.sort_values(["scenario", "return"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
