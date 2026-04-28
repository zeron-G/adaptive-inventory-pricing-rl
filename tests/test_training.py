from inventory_pricing_rl.agents import TrainingConfig, evaluate_policy, greedy_policy, train_tabular
from inventory_pricing_rl.data import make_synthetic_series
from inventory_pricing_rl.environment import EnvironmentConfig, InventoryPricingEnv
from inventory_pricing_rl.experiments import make_env_factory


def test_q_learning_trains_and_evaluates_on_synthetic_data():
    data, meta = make_synthetic_series(n_days=220)
    config = EnvironmentConfig.from_metadata(meta, episode_length=15)
    env_factory = make_env_factory(data, config)
    train_config = TrainingConfig(algorithm="q_learning", episodes=8, seed=3)
    q_table, history = train_tabular(env_factory, train_config)
    env = InventoryPricingEnv(data, config, split="train", seed=4)
    assert q_table.shape == (env.n_states, env.n_actions)
    assert len(history) == 8
    evaluation = evaluate_policy(env_factory, greedy_policy(q_table), split="validation", episodes=3, seed=5)
    assert len(evaluation) == 3
    assert evaluation["return"].notna().all()


def test_sarsa_trains_on_synthetic_data():
    data, meta = make_synthetic_series(n_days=220)
    config = EnvironmentConfig.from_metadata(meta, episode_length=15)
    env_factory = make_env_factory(data, config)
    train_config = TrainingConfig(algorithm="sarsa", episodes=8, seed=6)
    q_table, history = train_tabular(env_factory, train_config)
    assert q_table.ndim == 2
    assert history["algorithm"].unique().tolist() == ["sarsa"]
