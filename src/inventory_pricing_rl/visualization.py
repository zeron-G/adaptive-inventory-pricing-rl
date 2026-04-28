"""Plotting utilities for experiment reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_data_overview(daily_data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(daily_data["date"], daily_data["sales"], color="#2364aa", linewidth=0.9)
    axes[0].set_ylabel("Units sold")
    axes[0].set_title("Selected SKU daily sales")
    axes[1].plot(daily_data["date"], daily_data["sell_price"], color="#3da35d", linewidth=1.0)
    axes[1].set_ylabel("Sell price")
    axes[1].set_title("Observed sell price")
    axes[2].plot(daily_data["date"], daily_data["rolling_demand_7"], color="#f18f01", linewidth=1.0)
    axes[2].set_ylabel("7-day signal")
    axes[2].set_title("Lagged rolling demand signal")
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / "data_overview.png", dpi=170)
    plt.close(fig)


def save_training_curve(training_history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for algorithm, group in training_history.groupby("algorithm"):
        smooth = group["return"].rolling(50, min_periods=1).mean()
        ax.plot(group["episode"], smooth, label=algorithm.replace("_", " ").title(), linewidth=1.8)
    ax.set_title("Training learning curve (50-episode rolling mean)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Training return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=170)
    plt.close(fig)


def save_evaluation_charts(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("return_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(ordered["policy"], ordered["return_mean"], xerr=ordered["return_std"], color="#2364aa", alpha=0.85)
    ax.set_title("Average test return by policy")
    ax.set_xlabel("Mean cumulative reward")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_returns.png", dpi=170)
    plt.close(fig)

    metrics = ["fill_rate_mean", "stockout_units_mean", "avg_inventory_mean", "price_changes_mean"]
    labels = ["Fill rate", "Stockout units", "Average inventory", "Price changes"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, label in zip(axes.ravel(), metrics, labels, strict=True):
        ordered_metric = summary.sort_values(metric, ascending=True)
        ax.barh(ordered_metric["policy"], ordered_metric[metric], color="#3da35d", alpha=0.85)
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_operational_metrics.png", dpi=170)
    plt.close(fig)


def save_policy_heatmap(policy_table: pd.DataFrame, output_dir: Path, policy_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = policy_table[
        (policy_table["calendar_type"] == "weekday")
        & (policy_table["pipeline_bin"] == "none")
        & (policy_table["price_tier"] == "regular")
    ].copy()
    pivot = subset.pivot_table(
        index="inventory_bin",
        columns="demand_signal",
        values="action",
        aggfunc="first",
    )
    inventory_order = ["stockout", "low", "medium", "high", "excess"]
    demand_order = ["low", "normal", "high"]
    pivot = pivot.reindex(index=inventory_order, columns=demand_order)
    label_to_code = {label: idx for idx, label in enumerate(sorted(policy_table["action"].unique()))}
    coded = pivot.replace(label_to_code).astype(float)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    image = ax.imshow(coded.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(demand_order)), demand_order)
    ax.set_yticks(np.arange(len(inventory_order)), inventory_order)
    for i in range(coded.shape[0]):
        for j in range(coded.shape[1]):
            label = pivot.iloc[i, j]
            ax.text(j, i, str(label).replace(" / ", "\n"), ha="center", va="center", color="white", fontsize=8)
    ax.set_title(f"{policy_name} policy slice: weekday, regular price, no pipeline")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Action code")
    fig.tight_layout()
    fig.savefig(output_dir / f"{policy_name.lower().replace(' ', '_')}_policy_heatmap.png", dpi=170)
    plt.close(fig)


def create_all_plots(reports_dir: Path, daily_data: pd.DataFrame) -> None:
    figures = reports_dir / "figures"
    tables = reports_dir / "tables"
    save_data_overview(daily_data, figures)
    save_training_curve(pd.read_csv(tables / "training_history.csv"), figures)
    save_evaluation_charts(pd.read_csv(tables / "evaluation_summary.csv"), figures)
    for algorithm in ["q_learning", "sarsa"]:
        path = tables / f"{algorithm}_policy_table.csv"
        if path.exists():
            save_policy_heatmap(pd.read_csv(path), figures, algorithm.replace("_", " ").title())
