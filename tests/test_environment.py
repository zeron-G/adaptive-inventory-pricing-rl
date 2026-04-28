from inventory_pricing_rl.data import make_synthetic_series
from inventory_pricing_rl.environment import EnvironmentConfig, InventoryPricingEnv


def test_environment_state_and_step_are_finite():
    data, meta = make_synthetic_series(n_days=180)
    config = EnvironmentConfig.from_metadata(meta, episode_length=10)
    env = InventoryPricingEnv(data, config, split="train", seed=1)
    state = env.reset()
    assert 0 <= state < env.n_states
    next_state, reward, done, info = env.step(0)
    assert 0 <= next_state < env.n_states
    assert isinstance(reward, float)
    assert not done
    assert {"demand", "fulfilled_demand", "ending_inventory", "lost_sales"}.issubset(info)


def test_action_labels_cover_all_actions():
    data, meta = make_synthetic_series(n_days=180)
    config = EnvironmentConfig.from_metadata(meta, episode_length=5)
    env = InventoryPricingEnv(data, config, split="train", seed=2)
    labels = [env.action_label(i) for i in range(env.n_actions)]
    assert env.n_actions == 12
    assert len(set(labels)) == 12
