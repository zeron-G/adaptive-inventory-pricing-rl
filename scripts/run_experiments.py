"""Run preprocessing, tabular RL tuning/training, evaluation, and plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from inventory_pricing_rl.experiments import run_full_experiment
from inventory_pricing_rl.visualization import create_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/m5")
    parser.add_argument("--processed-dir", default="data/processed/m5_subset")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data for CI/smoke runs.")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--tuning-episodes", type=int, default=600)
    parser.add_argument("--final-episodes", type=int, default=3000)
    parser.add_argument("--evaluation-episodes", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_full_experiment(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        reports_dir=Path(args.reports_dir),
        use_synthetic=args.synthetic,
        force_prepare=args.force_prepare,
        tuning_episodes=args.tuning_episodes,
        final_episodes=args.final_episodes,
        evaluation_episodes=args.evaluation_episodes,
    )
    create_all_plots(Path(args.reports_dir), result["daily_data"])
    print("Experiment complete.")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
