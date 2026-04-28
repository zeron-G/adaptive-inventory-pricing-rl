"""End-to-end experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inventory_pricing_rl.agents import TrainingConfig, evaluate_policy, greedy_policy, train_tabular
from inventory_pricing_rl.baselines import BASELINE_POLICIES
from inventory_pricing_rl.data import SelectedSkuMetadata, build_selected_series, load_processed_series, make_synthetic_series
from inventory_pricing_rl.environment import EnvironmentConfig, InventoryPricingEnv


def make_env_factory(
    daily_data: pd.DataFrame,
    env_config: EnvironmentConfig,
) -> Any:
    def factory(split: str = "train", seed: int | None = None) -> InventoryPricingEnv:
        return InventoryPricingEnv(daily_data, env_config, split=split, seed=seed)

    return factory


def summarize_evaluation(evaluation: pd.DataFrame) -> pd.DataFrame:
    metrics = ["return", "fill_rate", "stockout_units", "avg_inventory", "avg_order_quantity", "price_changes"]
    summary = (
        evaluation.groupby("policy")[metrics]
        .agg(["mean", "std"])
        .sort_values(("return", "mean"), ascending=False)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def tune_hyperparameters(
    env_factory: Any,
    algorithms: list[str],
    search_space: dict[str, list[float]],
    episodes: int,
    seeds: list[int],
    validation_episodes: int = 20,
) -> tuple[pd.DataFrame, dict[str, TrainingConfig]]:
    """Grid-search tabular RL hyperparameters on validation episodes."""

    records: list[dict[str, Any]] = []
    for algorithm in algorithms:
        keys = ["alpha", "gamma", "epsilon_decay", "epsilon_min"]
        for values in product(*(search_space[key] for key in keys)):
            params = dict(zip(keys, values, strict=True))
            validation_returns = []
            validation_fill = []
            for seed in seeds:
                cfg = TrainingConfig(algorithm=algorithm, episodes=episodes, seed=seed, **params)
                q_table, history = train_tabular(env_factory, cfg)
                eval_df = evaluate_policy(
                    env_factory,
                    greedy_policy(q_table),
                    split="validation",
                    episodes=validation_episodes,
                    seed=seed + 1000,
                    label=algorithm,
                )
                validation_returns.append(float(eval_df["return"].mean()))
                validation_fill.append(float(eval_df["fill_rate"].mean()))
            records.append(
                {
                    "algorithm": algorithm,
                    **params,
                    "episodes": episodes,
                    "seeds": len(seeds),
                    "validation_return_mean": float(np.mean(validation_returns)),
                    "validation_return_std": float(np.std(validation_returns, ddof=1)) if len(seeds) > 1 else 0.0,
                    "validation_fill_rate_mean": float(np.mean(validation_fill)),
                }
            )
    results = pd.DataFrame(records).sort_values("validation_return_mean", ascending=False).reset_index(drop=True)
    best: dict[str, TrainingConfig] = {}
    for algorithm in algorithms:
        row = results[results["algorithm"] == algorithm].iloc[0].to_dict()
        best[algorithm] = TrainingConfig(
            algorithm=algorithm,
            alpha=float(row["alpha"]),
            gamma=float(row["gamma"]),
            epsilon_decay=float(row["epsilon_decay"]),
            epsilon_min=float(row["epsilon_min"]),
            episodes=episodes,
            seed=seeds[0],
        )
    return results, best


def policy_table(env: InventoryPricingEnv, policy: np.ndarray, max_rows: int = 60) -> pd.DataFrame:
    rows = []
    for state_id, action in enumerate(policy):
        decoded = env.decode_state(state_id)
        rows.append({"state_id": state_id, **decoded, "action_id": int(action), "action": env.action_label(int(action))})
    return pd.DataFrame(rows).head(max_rows)


def run_full_experiment(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    use_synthetic: bool = False,
    force_prepare: bool = False,
    tuning_episodes: int = 600,
    final_episodes: int = 3000,
    evaluation_episodes: int = 120,
) -> dict[str, Any]:
    """Run preprocessing, hyperparameter tuning, final training, and evaluation."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = reports_dir / "tables"
    models_dir = reports_dir / "models"
    figures_dir = reports_dir / "figures"
    for directory in [tables_dir, models_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if use_synthetic:
        daily_data, meta = make_synthetic_series()
    else:
        if (processed_dir / "selected_sku_daily.csv").exists() and not force_prepare:
            daily_data, meta = load_processed_series(processed_dir)
        else:
            daily_data, meta = build_selected_series(raw_dir, processed_dir, force=force_prepare)

    env_config = EnvironmentConfig.from_metadata(meta)
    env_factory = make_env_factory(daily_data, env_config)
    probe_env = env_factory("train", 7)

    data_summary = {
        "metadata": asdict(meta),
        "environment_config": asdict(env_config),
        "n_states": probe_env.n_states,
        "n_actions": probe_env.n_actions,
        "state_action_values": probe_env.n_states * probe_env.n_actions,
        "train_rows": int((daily_data["split"] == "train").sum()),
        "validation_rows": int((daily_data["split"] == "validation").sum()),
        "test_rows": int((daily_data["split"] == "test").sum()),
    }
    (reports_dir / "experiment_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
    daily_data.groupby("split")["sales"].agg(["count", "mean", "std", "min", "median", "max"]).to_csv(
        tables_dir / "sales_by_split.csv"
    )
    daily_data[["date", "sales", "sell_price", "calendar_type", "demand_signal", "split"]].to_csv(
        tables_dir / "selected_series_preview.csv", index=False
    )

    if use_synthetic and tuning_episodes <= 20:
        search_space = {
            "alpha": [0.10],
            "gamma": [0.95],
            "epsilon_decay": [0.996],
            "epsilon_min": [0.05],
        }
        tuning_seeds = [13]
        validation_episodes = 3
    else:
        search_space = {
            "alpha": [0.05, 0.10, 0.20],
            "gamma": [0.90, 0.95, 0.99],
            "epsilon_decay": [0.992, 0.996, 0.999],
            "epsilon_min": [0.03, 0.05],
        }
        tuning_seeds = [13, 29, 47]
        validation_episodes = 20
    tuning_results, best_configs = tune_hyperparameters(
        env_factory,
        algorithms=["q_learning", "sarsa"],
        search_space=search_space,
        episodes=tuning_episodes,
        seeds=tuning_seeds,
        validation_episodes=validation_episodes,
    )
    tuning_results.to_csv(tables_dir / "hyperparameter_tuning_results.csv", index=False)

    all_training = []
    learned_policies: dict[str, np.ndarray] = {}
    for algorithm, cfg in best_configs.items():
        final_cfg = TrainingConfig(
            algorithm=algorithm,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            epsilon_decay=cfg.epsilon_decay,
            epsilon_min=cfg.epsilon_min,
            episodes=final_episodes,
            seed=2026,
        )
        q_table, history = train_tabular(env_factory, final_cfg)
        all_training.append(history)
        learned_policies[algorithm] = greedy_policy(q_table)
        np.save(models_dir / f"{algorithm}_q_table.npy", q_table)
        (models_dir / f"{algorithm}_training_config.json").write_text(
            json.dumps(asdict(final_cfg), indent=2), encoding="utf-8"
        )
        policy_table(probe_env, learned_policies[algorithm], max_rows=probe_env.n_states).to_csv(
            tables_dir / f"{algorithm}_policy_table.csv", index=False
        )

    training_history = pd.concat(all_training, ignore_index=True)
    training_history.to_csv(tables_dir / "training_history.csv", index=False)

    eval_frames = []
    for algorithm, policy in learned_policies.items():
        eval_frames.append(
            evaluate_policy(
                env_factory,
                policy,
                split="test",
                episodes=evaluation_episodes,
                seed=9000,
                label=algorithm.replace("_", " ").title(),
            )
        )
    for label, policy_fn in BASELINE_POLICIES.items():
        eval_frames.append(
            evaluate_policy(
                env_factory,
                policy_fn,
                split="test",
                episodes=evaluation_episodes,
                seed=9100,
                label=label,
            )
        )
    evaluation = pd.concat(eval_frames, ignore_index=True)
    evaluation.to_csv(tables_dir / "evaluation_episodes.csv", index=False)
    summary = summarize_evaluation(evaluation)
    summary.to_csv(tables_dir / "evaluation_summary.csv", index=False)

    return {
        "daily_data": daily_data,
        "metadata": meta,
        "env_config": env_config,
        "tuning_results": tuning_results,
        "training_history": training_history,
        "evaluation": evaluation,
        "summary": summary,
        "reports_dir": reports_dir,
    }
