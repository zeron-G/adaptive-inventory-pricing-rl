"""Data loading and M5 preprocessing utilities.

The project uses M5 as a demand-calibration source, not as a complete
inventory-transition dataset. M5 reports observed unit sales and sell prices,
but it does not contain on-hand inventory, true lost demand, or replenishment
orders. The simulator therefore makes those operational dynamics explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import zipfile

import numpy as np
import pandas as pd


M5_ZENODO_URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"


@dataclass(frozen=True)
class SelectedSkuMetadata:
    """Metadata for the SKU-store series selected for experiments."""

    item_id: str
    store_id: str
    state_id: str
    dept_id: str
    cat_id: str
    mean_sales: float
    nonzero_ratio: float
    price_cv: float
    median_price: float
    capacity: int
    unit_cost: float
    small_order: int
    medium_order: int
    large_order: int
    train_days: int
    validation_days: int
    test_days: int


def ensure_m5_extracted(raw_dir: Path) -> None:
    """Extract the M5 zip file when the individual CSVs are missing."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "m5-forecasting-accuracy.zip"
    expected = raw_dir / "calendar.csv"
    if expected.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Download it from {M5_ZENODO_URL} or run scripts/download_m5.py."
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


def _day_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("d_")]


def _calendar_type(row: pd.Series, state_id: str) -> str:
    snap_col = f"snap_{state_id}"
    has_event = pd.notna(row.get("event_name_1")) or pd.notna(row.get("event_name_2"))
    is_snap = int(row.get(snap_col, 0)) == 1 if snap_col in row else False
    if has_event or is_snap:
        return "event_or_snap"
    if int(row["wday"]) in (1, 2):
        return "weekend"
    return "weekday"


def select_sku(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    min_nonzero_ratio: float = 0.45,
    preferred_max_mean_sales: float = 30.0,
) -> pd.Series:
    """Choose a SKU-store row with rich enough demand and price variation.

    The score favors stable but nontrivial demand and observed price variation.
    This avoids a degenerate series where all methods learn simply not to order.
    """

    days = _day_columns(sales)
    metrics = sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    values = sales[days]
    metrics["mean_sales"] = values.mean(axis=1)
    metrics["nonzero_ratio"] = (values > 0).mean(axis=1)

    price_stats = (
        prices.groupby(["item_id", "store_id"])["sell_price"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={"mean": "price_mean", "std": "price_std", "median": "median_price"})
    )
    price_stats["price_cv"] = price_stats["price_std"].fillna(0) / price_stats["price_mean"].replace(0, np.nan)
    metrics = metrics.merge(price_stats, on=["item_id", "store_id"], how="left")
    metrics["price_cv"] = metrics["price_cv"].fillna(0)
    metrics["median_price"] = metrics["median_price"].fillna(metrics["price_mean"]).fillna(1.0)

    candidate = metrics[
        (metrics["nonzero_ratio"] >= min_nonzero_ratio)
        & (metrics["mean_sales"] >= 1.0)
        & (metrics["mean_sales"] <= preferred_max_mean_sales)
        & (metrics["median_price"] > 0)
    ].copy()
    if candidate.empty:
        candidate = metrics[
            (metrics["nonzero_ratio"] >= min_nonzero_ratio)
            & (metrics["mean_sales"] >= 1.0)
            & (metrics["mean_sales"] <= preferred_max_mean_sales * 3)
            & (metrics["median_price"] > 0)
        ].copy()
    if candidate.empty:
        candidate = metrics[(metrics["mean_sales"] >= 0.5) & (metrics["median_price"] > 0)].copy()

    candidate["score"] = (
        np.log1p(candidate["mean_sales"]) * 0.55
        + candidate["nonzero_ratio"] * 0.30
        + candidate["price_cv"].clip(0, 0.50) * 0.15
    )
    best_idx = candidate["score"].idxmax()
    return sales.loc[best_idx]


def build_selected_series(raw_dir: Path, output_dir: Path, force: bool = False) -> tuple[pd.DataFrame, SelectedSkuMetadata]:
    """Prepare one SKU-store daily series and write processed artifacts.

    Returns the processed daily dataframe and metadata. The dataframe includes
    observed sales, sell price, calendar context, rolling demand signal, and
    chronological split labels. Inventory is intentionally not fabricated here;
    it is owned by the simulator.
    """

    ensure_m5_extracted(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "selected_sku_daily.csv"
    out_meta = output_dir / "selected_sku_metadata.json"
    if out_csv.exists() and out_meta.exists() and not force:
        df = pd.read_csv(out_csv, parse_dates=["date"])
        meta = SelectedSkuMetadata(**json.loads(out_meta.read_text(encoding="utf-8")))
        return df, meta

    sales_path = raw_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        sales_path = raw_dir / "sales_train_validation.csv"
    sales = pd.read_csv(sales_path)
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])

    selected = select_sku(sales, prices)
    days = _day_columns(sales)
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    selected_df = selected[id_vars + days].to_frame().T
    long_df = selected_df.melt(id_vars=id_vars, value_vars=days, var_name="d", value_name="sales")
    long_df["sales"] = long_df["sales"].astype(int)
    long_df = long_df.merge(calendar, on="d", how="left")
    long_df = long_df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    long_df["sell_price"] = long_df["sell_price"].ffill().bfill()
    long_df["calendar_type"] = long_df.apply(lambda row: _calendar_type(row, selected["state_id"]), axis=1)

    # Use shifted rolling demand so the signal at day t only uses information
    # that would have been available before day t demand is realized.
    long_df["rolling_demand_7"] = (
        long_df["sales"].shift(1).rolling(7, min_periods=1).mean().fillna(long_df["sales"].expanding().mean())
    )
    q1, q2 = long_df["rolling_demand_7"].quantile([1 / 3, 2 / 3]).to_numpy()
    long_df["demand_signal"] = pd.cut(
        long_df["rolling_demand_7"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "normal", "high"],
        include_lowest=True,
    ).astype(str)

    median_price = float(long_df["sell_price"].median())
    long_df["observed_price_tier"] = pd.cut(
        long_df["sell_price"] / median_price,
        bins=[-np.inf, 0.95, 1.05, np.inf],
        labels=["discount", "regular", "premium"],
    ).astype(str)

    n = len(long_df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    long_df["split"] = "test"
    long_df.loc[: train_end - 1, "split"] = "train"
    long_df.loc[train_end : validation_end - 1, "split"] = "validation"

    demand_q95 = float(long_df.loc[long_df["split"] == "train", "sales"].quantile(0.95))
    mean_sales = float(long_df["sales"].mean())
    capacity = int(max(30, math.ceil(max(demand_q95 * 5, mean_sales * 10))))
    capacity = min(capacity, 140)
    small_order = max(1, int(round(capacity * 0.15)))
    medium_order = max(small_order + 1, int(round(capacity * 0.30)))
    large_order = max(medium_order + 1, int(round(capacity * 0.50)))

    selected_prices = prices[(prices["item_id"] == selected["item_id"]) & (prices["store_id"] == selected["store_id"])]
    price_cv = float(selected_prices["sell_price"].std() / selected_prices["sell_price"].mean())
    meta = SelectedSkuMetadata(
        item_id=str(selected["item_id"]),
        store_id=str(selected["store_id"]),
        state_id=str(selected["state_id"]),
        dept_id=str(selected["dept_id"]),
        cat_id=str(selected["cat_id"]),
        mean_sales=float(long_df["sales"].mean()),
        nonzero_ratio=float((long_df["sales"] > 0).mean()),
        price_cv=price_cv,
        median_price=median_price,
        capacity=capacity,
        unit_cost=float(median_price * 0.62),
        small_order=small_order,
        medium_order=medium_order,
        large_order=large_order,
        train_days=int((long_df["split"] == "train").sum()),
        validation_days=int((long_df["split"] == "validation").sum()),
        test_days=int((long_df["split"] == "test").sum()),
    )

    long_df.to_csv(out_csv, index=False)
    out_meta.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return long_df, meta


def load_processed_series(processed_dir: Path) -> tuple[pd.DataFrame, SelectedSkuMetadata]:
    """Load processed data produced by build_selected_series."""

    df = pd.read_csv(processed_dir / "selected_sku_daily.csv", parse_dates=["date"])
    meta = SelectedSkuMetadata(**json.loads((processed_dir / "selected_sku_metadata.json").read_text(encoding="utf-8")))
    return df, meta


def make_synthetic_series(n_days: int = 365, seed: int = 7) -> tuple[pd.DataFrame, SelectedSkuMetadata]:
    """Create a small deterministic dataset for tests and CI smoke runs."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    base = 4 + 1.2 * (dates.dayofweek >= 5).astype(float) + 0.8 * np.sin(np.arange(n_days) / 14)
    sales = rng.poisson(np.clip(base, 0.5, None))
    price = 3.0 + 0.15 * np.sin(np.arange(n_days) / 20)
    df = pd.DataFrame(
        {
            "date": dates,
            "d": [f"d_{i+1}" for i in range(n_days)],
            "item_id": "SYNTH_001",
            "dept_id": "SYNTH_DEPT",
            "cat_id": "SYNTH",
            "store_id": "SYNTH_STORE",
            "state_id": "CA",
            "sales": sales,
            "sell_price": price,
            "wday": dates.dayofweek + 1,
            "event_name_1": np.nan,
            "event_name_2": np.nan,
            "snap_CA": 0,
            "calendar_type": np.where(dates.dayofweek >= 5, "weekend", "weekday"),
        }
    )
    rolling = pd.Series(sales).shift(1).rolling(7, min_periods=1).mean().fillna(np.mean(sales))
    q1, q2 = rolling.quantile([1 / 3, 2 / 3]).to_numpy()
    df["rolling_demand_7"] = rolling
    df["demand_signal"] = pd.cut(
        rolling,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "normal", "high"],
        include_lowest=True,
    ).astype(str)
    df["observed_price_tier"] = "regular"
    train_end = int(n_days * 0.70)
    val_end = int(n_days * 0.85)
    df["split"] = "test"
    df.loc[: train_end - 1, "split"] = "train"
    df.loc[train_end : val_end - 1, "split"] = "validation"
    meta = SelectedSkuMetadata(
        item_id="SYNTH_001",
        store_id="SYNTH_STORE",
        state_id="CA",
        dept_id="SYNTH_DEPT",
        cat_id="SYNTH",
        mean_sales=float(np.mean(sales)),
        nonzero_ratio=float(np.mean(sales > 0)),
        price_cv=float(np.std(price) / np.mean(price)),
        median_price=float(np.median(price)),
        capacity=50,
        unit_cost=1.85,
        small_order=8,
        medium_order=15,
        large_order=25,
        train_days=train_end,
        validation_days=val_end - train_end,
        test_days=n_days - val_end,
    )
    return df, meta
