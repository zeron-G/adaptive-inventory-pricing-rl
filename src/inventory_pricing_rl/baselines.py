"""Interpretable non-RL baseline policies."""

from __future__ import annotations

import numpy as np

from inventory_pricing_rl.environment import (
    DEMAND_BINS,
    INVENTORY_BINS,
    ORDER_LABELS,
    PRICE_TIERS,
    InventoryPricingEnv,
)


def action_id(env: InventoryPricingEnv, price_label: str, order_label: str) -> int:
    price_idx = PRICE_TIERS.index(price_label)
    order_idx = ORDER_LABELS.index(order_label)
    return env.actions.index((price_idx, order_idx))


def random_policy(env: InventoryPricingEnv, state: int) -> int:
    return int(env.rng.integers(env.n_actions))


def static_reorder_point_policy(env: InventoryPricingEnv, state: int) -> int:
    decoded = env.decode_state(state)
    inventory_position = env.inventory + sum(env.pipeline)
    reorder_point = int(env.config.capacity * 0.35)
    target = int(env.config.capacity * 0.70)
    if inventory_position <= reorder_point:
        gap = target - inventory_position
        if gap >= env.config.order_quantities[3] * 0.75:
            order = "large"
        elif gap >= env.config.order_quantities[2] * 0.75:
            order = "medium"
        else:
            order = "small"
    else:
        order = "none"
    return action_id(env, "regular", order)


def average_demand_order_up_to_policy(env: InventoryPricingEnv, state: int) -> int:
    train_mean = float(env.daily_data.loc[env.daily_data["split"] == "train", "sales"].mean())
    target = int(train_mean * (env.config.lead_time + 5))
    inventory_position = env.inventory + sum(env.pipeline)
    gap = max(0, target - inventory_position)
    if gap <= 0:
        order = "none"
    elif gap <= env.config.order_quantities[1]:
        order = "small"
    elif gap <= env.config.order_quantities[2]:
        order = "medium"
    else:
        order = "large"
    return action_id(env, "regular", order)


def markdown_inventory_policy(env: InventoryPricingEnv, state: int) -> int:
    decoded = env.decode_state(state)
    inventory_bin = decoded["inventory_bin"]
    demand_bin = decoded["demand_signal"]

    if inventory_bin in {"high", "excess"} and demand_bin in {"low", "normal"}:
        price = "discount"
    elif inventory_bin in {"stockout", "low"} or demand_bin == "high":
        price = "premium"
    else:
        price = "regular"

    if inventory_bin in {"stockout", "low"} and decoded["pipeline_bin"] == "none":
        order = "large" if demand_bin == "high" else "medium"
    elif inventory_bin == "medium" and demand_bin == "high":
        order = "small"
    else:
        order = "none"
    return action_id(env, price, order)


BASELINE_POLICIES = {
    "Random valid action": random_policy,
    "Static regular + reorder point": static_reorder_point_policy,
    "Regular + average demand order-up-to": average_demand_order_up_to_policy,
    "Inventory markdown heuristic": markdown_inventory_policy,
}
