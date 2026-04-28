"""Finite MDP simulator for joint inventory and pricing decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from inventory_pricing_rl.data import SelectedSkuMetadata


INVENTORY_BINS = ["stockout", "low", "medium", "high", "excess"]
DEMAND_BINS = ["low", "normal", "high"]
PRICE_TIERS = ["discount", "regular", "premium"]
CALENDAR_TYPES = ["weekday", "weekend", "event_or_snap"]
PIPELINE_BINS = ["none", "small", "large"]
ORDER_LABELS = ["none", "small", "medium", "large"]


@dataclass(frozen=True)
class EnvironmentConfig:
    """Operational assumptions for the simulator."""

    capacity: int
    median_price: float
    unit_cost: float
    order_quantities: tuple[int, int, int, int]
    lead_time: int = 2
    holding_cost_rate: float = 0.015
    stockout_penalty_rate: float = 0.70
    fixed_order_cost: float = 1.50
    excess_penalty_rate: float = 0.01
    price_change_penalty: float = 0.05
    discount_multiplier: float = 0.90
    premium_multiplier: float = 1.10
    discount_demand_lift: float = 0.15
    premium_demand_drop: float = 0.15
    episode_length: int = 120
    initial_inventory_ratio: float = 0.50
    random_start: bool = True

    @classmethod
    def from_metadata(cls, meta: SelectedSkuMetadata, **overrides: Any) -> "EnvironmentConfig":
        params = {
            "capacity": meta.capacity,
            "median_price": meta.median_price,
            "unit_cost": meta.unit_cost,
            "order_quantities": (0, meta.small_order, meta.medium_order, meta.large_order),
        }
        params.update(overrides)
        return cls(**params)


class InventoryPricingEnv:
    """A small tabular retail simulator calibrated from daily sales data.

    The environment intentionally exposes a compact finite state. This keeps the
    project aligned with tabular RL and makes learned policies inspectable.
    """

    def __init__(
        self,
        daily_data: pd.DataFrame,
        config: EnvironmentConfig,
        split: str = "train",
        seed: int | None = None,
    ) -> None:
        self.daily_data = daily_data.reset_index(drop=True).copy()
        self.config = config
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.split_data = self.daily_data[self.daily_data["split"] == split].reset_index(drop=True)
        if self.split_data.empty:
            raise ValueError(f"No rows for split={split!r}.")

        self.price_values = np.array(
            [
                config.median_price * config.discount_multiplier,
                config.median_price,
                config.median_price * config.premium_multiplier,
            ],
            dtype=float,
        )
        self.price_demand_factors = np.array(
            [1.0 + config.discount_demand_lift, 1.0, max(0.05, 1.0 - config.premium_demand_drop)],
            dtype=float,
        )
        self.actions = [(p, q) for p in range(len(PRICE_TIERS)) for q in range(len(ORDER_LABELS))]
        self.n_actions = len(self.actions)
        self.state_shape = (
            len(INVENTORY_BINS),
            len(DEMAND_BINS),
            len(PRICE_TIERS),
            len(CALENDAR_TYPES),
            len(PIPELINE_BINS),
        )
        self.n_states = int(np.prod(self.state_shape))

        self._build_demand_lookup()
        self.reset()

    def _build_demand_lookup(self) -> None:
        train = self.daily_data[self.daily_data["split"] == "train"].copy()
        if train.empty:
            train = self.daily_data.copy()
        self.train_demand_q1, self.train_demand_q2 = train["rolling_demand_7"].quantile([1 / 3, 2 / 3]).to_numpy()
        self.global_sales = train["sales"].astype(int).to_numpy()
        self.lookup: dict[tuple[str, str], np.ndarray] = {}
        for (cal, dem), group in train.groupby(["calendar_type", "demand_signal"]):
            samples = group["sales"].astype(int).to_numpy()
            if len(samples) >= 5:
                self.lookup[(str(cal), str(dem))] = samples
        self.calendar_lookup: dict[str, np.ndarray] = {}
        for cal, group in train.groupby("calendar_type"):
            samples = group["sales"].astype(int).to_numpy()
            if len(samples) >= 5:
                self.calendar_lookup[str(cal)] = samples

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = max(0, len(self.split_data) - self.config.episode_length)
        self.day_idx = int(self.rng.integers(0, max_start + 1)) if self.config.random_start and max_start > 0 else 0
        self.steps = 0
        self.inventory = int(round(self.config.capacity * self.config.initial_inventory_ratio))
        self.pipeline = [0 for _ in range(self.config.lead_time)]
        self.current_price_tier = PRICE_TIERS.index("regular")
        first_signal = str(self.split_data.loc[self.day_idx, "demand_signal"])
        self.recent_demands = [float(self.split_data["sales"].mean()) for _ in range(7)]
        self.current_demand_signal = first_signal if first_signal in DEMAND_BINS else "normal"
        return self._state_id()

    def _inventory_bin(self) -> str:
        ratio = self.inventory / max(1, self.config.capacity)
        if self.inventory <= 0:
            return "stockout"
        if ratio <= 0.20:
            return "low"
        if ratio <= 0.55:
            return "medium"
        if ratio <= 0.80:
            return "high"
        return "excess"

    def _pipeline_bin(self) -> str:
        total = sum(self.pipeline)
        if total <= 0:
            return "none"
        if total <= self.config.order_quantities[2]:
            return "small"
        return "large"

    def _calendar_type(self) -> str:
        value = str(self.split_data.loc[self.day_idx, "calendar_type"])
        return value if value in CALENDAR_TYPES else "weekday"

    def _state_tuple(self) -> tuple[int, int, int, int, int]:
        inv = INVENTORY_BINS.index(self._inventory_bin())
        demand = DEMAND_BINS.index(self.current_demand_signal if self.current_demand_signal in DEMAND_BINS else "normal")
        price = self.current_price_tier
        cal = CALENDAR_TYPES.index(self._calendar_type())
        pipe = PIPELINE_BINS.index(self._pipeline_bin())
        return inv, demand, price, cal, pipe

    def _state_id(self) -> int:
        return int(np.ravel_multi_index(self._state_tuple(), self.state_shape))

    def decode_state(self, state_id: int) -> dict[str, str]:
        inv, demand, price, cal, pipe = np.unravel_index(state_id, self.state_shape)
        return {
            "inventory_bin": INVENTORY_BINS[int(inv)],
            "demand_signal": DEMAND_BINS[int(demand)],
            "price_tier": PRICE_TIERS[int(price)],
            "calendar_type": CALENDAR_TYPES[int(cal)],
            "pipeline_bin": PIPELINE_BINS[int(pipe)],
        }

    def action_label(self, action_id: int) -> str:
        price_idx, order_idx = self.actions[action_id]
        return f"{PRICE_TIERS[price_idx]} / {ORDER_LABELS[order_idx]}"

    def _sample_base_demand(self) -> int:
        cal = self._calendar_type()
        dem = self.current_demand_signal if self.current_demand_signal in DEMAND_BINS else "normal"
        samples = self.lookup.get((cal, dem))
        if samples is None:
            samples = self.calendar_lookup.get(cal, self.global_sales)
        return int(self.rng.choice(samples))

    def _update_demand_signal(self, realized_demand: int) -> None:
        self.recent_demands.append(float(realized_demand))
        self.recent_demands = self.recent_demands[-7:]
        rolling = float(np.mean(self.recent_demands))
        if rolling <= self.train_demand_q1:
            self.current_demand_signal = "low"
        elif rolling <= self.train_demand_q2:
            self.current_demand_signal = "normal"
        else:
            self.current_demand_signal = "high"

    def step(self, action_id: int) -> tuple[int, float, bool, dict[str, float]]:
        price_idx, order_idx = self.actions[action_id]

        arrivals = self.pipeline.pop(0) if self.pipeline else 0
        self.inventory = min(self.config.capacity, self.inventory + arrivals)

        base_demand = self._sample_base_demand()
        demand_lambda = max(0.05, base_demand * self.price_demand_factors[price_idx])
        demand = int(self.rng.poisson(demand_lambda))
        fulfilled = min(self.inventory, demand)
        lost_sales = max(0, demand - fulfilled)
        revenue = self.price_values[price_idx] * fulfilled

        self.inventory -= fulfilled
        order_qty = int(self.config.order_quantities[order_idx])
        if self.config.lead_time > 0:
            self.pipeline.append(order_qty)
        else:
            self.inventory = min(self.config.capacity, self.inventory + order_qty)

        procurement = self.config.unit_cost * order_qty
        fixed_order = self.config.fixed_order_cost if order_qty > 0 else 0.0
        holding = self.config.holding_cost_rate * self.config.unit_cost * self.inventory
        stockout_penalty = self.config.stockout_penalty_rate * self.config.median_price * lost_sales
        excess_units = max(0, self.inventory - int(self.config.capacity * 0.80))
        excess_penalty = self.config.excess_penalty_rate * self.config.unit_cost * excess_units
        price_change_penalty = self.config.price_change_penalty if price_idx != self.current_price_tier else 0.0

        reward = revenue - procurement - fixed_order - holding - stockout_penalty - excess_penalty - price_change_penalty

        self.current_price_tier = price_idx
        self._update_demand_signal(demand)
        self.steps += 1
        self.day_idx = (self.day_idx + 1) % len(self.split_data)
        done = self.steps >= self.config.episode_length
        info = {
            "reward": float(reward),
            "revenue": float(revenue),
            "procurement": float(procurement),
            "holding": float(holding),
            "stockout_penalty": float(stockout_penalty),
            "lost_sales": float(lost_sales),
            "fulfilled_demand": float(fulfilled),
            "demand": float(demand),
            "ending_inventory": float(self.inventory),
            "order_quantity": float(order_qty),
            "price": float(self.price_values[price_idx]),
            "price_change": float(price_change_penalty > 0),
        }
        return self._state_id(), float(reward), done, info

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
