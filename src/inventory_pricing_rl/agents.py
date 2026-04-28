"""Tabular reinforcement-learning algorithms and evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from inventory_pricing_rl.environment import InventoryPricingEnv


EnvFactory = Callable[[str, int | None], InventoryPricingEnv]


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str
    alpha: float = 0.10
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 1000
    seed: int = 7


def epsilon_by_episode(config: TrainingConfig, episode: int) -> float:
    return max(config.epsilon_min, config.epsilon_start * (config.epsilon_decay**episode))


def choose_epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def train_tabular(env_factory: EnvFactory, config: TrainingConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Train Q-learning or SARSA and return the Q-table plus episode history."""

    algorithm = config.algorithm.lower()
    if algorithm not in {"q_learning", "sarsa"}:
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")

    rng = np.random.default_rng(config.seed)
    env = env_factory("train", config.seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    records: list[dict[str, float]] = []

    for episode in range(config.episodes):
        state = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        epsilon = epsilon_by_episode(config, episode)
        action = choose_epsilon_greedy(q_table, state, epsilon, rng)
        done = False
        total_reward = 0.0
        total_lost_sales = 0.0
        total_demand = 0.0
        total_fulfilled = 0.0
        total_inventory = 0.0
        steps = 0

        while not done:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            total_lost_sales += info["lost_sales"]
            total_demand += info["demand"]
            total_fulfilled += info["fulfilled_demand"]
            total_inventory += info["ending_inventory"]
            steps += 1

            if algorithm == "q_learning":
                target = reward if done else reward + config.gamma * np.max(q_table[next_state])
                next_action = choose_epsilon_greedy(q_table, next_state, epsilon, rng)
            else:
                next_action = choose_epsilon_greedy(q_table, next_state, epsilon, rng)
                target = reward if done else reward + config.gamma * q_table[next_state, next_action]

            q_table[state, action] += config.alpha * (target - q_table[state, action])
            state, action = next_state, next_action

        records.append(
            {
                "episode": episode,
                "algorithm": algorithm,
                "alpha": config.alpha,
                "gamma": config.gamma,
                "epsilon_min": config.epsilon_min,
                "epsilon_decay": config.epsilon_decay,
                "epsilon": epsilon,
                "return": total_reward,
                "fill_rate": total_fulfilled / max(total_demand, 1.0),
                "lost_sales": total_lost_sales,
                "avg_inventory": total_inventory / max(steps, 1),
            }
        )

    return q_table, pd.DataFrame(records)


def greedy_policy(q_table: np.ndarray) -> np.ndarray:
    return np.argmax(q_table, axis=1).astype(int)


def evaluate_policy(
    env_factory: EnvFactory,
    policy: np.ndarray | Callable[[InventoryPricingEnv, int], int],
    split: str = "test",
    episodes: int = 100,
    seed: int = 123,
    label: str = "policy",
) -> pd.DataFrame:
    """Evaluate a deterministic policy or callable baseline."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    env = env_factory(split, seed)
    for episode in range(episodes):
        state = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        totals = {
            "return": 0.0,
            "revenue": 0.0,
            "procurement": 0.0,
            "holding": 0.0,
            "stockout_penalty": 0.0,
            "lost_sales": 0.0,
            "demand": 0.0,
            "fulfilled_demand": 0.0,
            "ending_inventory": 0.0,
            "order_quantity": 0.0,
            "price_change": 0.0,
        }
        steps = 0
        while not done:
            if callable(policy):
                action = int(policy(env, state))
            else:
                action = int(policy[state])
            state, reward, done, info = env.step(action)
            totals["return"] += reward
            for key in totals:
                if key != "return":
                    totals[key] += info.get(key, 0.0)
            steps += 1
        rows.append(
            {
                "policy": label,
                "split": split,
                "episode": episode,
                "return": totals["return"],
                "revenue": totals["revenue"],
                "procurement": totals["procurement"],
                "holding": totals["holding"],
                "stockout_penalty": totals["stockout_penalty"],
                "fill_rate": totals["fulfilled_demand"] / max(totals["demand"], 1.0),
                "stockout_units": totals["lost_sales"],
                "avg_inventory": totals["ending_inventory"] / max(steps, 1),
                "avg_order_quantity": totals["order_quantity"] / max(steps, 1),
                "price_changes": totals["price_change"],
            }
        )
    return pd.DataFrame(rows)


def config_to_dict(config: TrainingConfig) -> dict[str, float | str | int]:
    return asdict(config)
