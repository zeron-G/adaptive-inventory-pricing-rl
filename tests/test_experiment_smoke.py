from pathlib import Path

from inventory_pricing_rl.experiments import run_full_experiment


def test_full_experiment_smoke(tmp_path: Path):
    result = run_full_experiment(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        use_synthetic=True,
        tuning_episodes=4,
        final_episodes=8,
        evaluation_episodes=3,
    )
    assert not result["summary"].empty
    assert (tmp_path / "reports" / "tables" / "evaluation_summary.csv").exists()
    assert (tmp_path / "reports" / "models" / "q_learning_q_table.npy").exists()
