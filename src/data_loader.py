from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"date", "equity_close", "gold_close"}


def load_price_data(filepath: str) -> pd.DataFrame:
    """Load prices, validate required columns, and return a dated clean frame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Price data was not found at '{path}'.")
    raw = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"prices.csv is missing required columns: {sorted(missing)}")
    df = raw.loc[:, ["date", "equity_close", "gold_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("equity_close", "gold_close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    df = df.sort_values("date").set_index("date").ffill().dropna()
    if len(df) < 80:
        raise ValueError("prices.csv needs at least 80 valid daily rows for the demo.")
    if (df[["equity_close", "gold_close"]] <= 0).any().any():
        raise ValueError("Price columns must contain positive values.")
    return df


def label_crash_windows(df: pd.DataFrame, drawdown_threshold: float = -0.10, lookahead_days: int = 30) -> pd.Series:
    """Label dates whose trailing peak will suffer a drawdown within the lookahead."""
    # A rolling peak resets after a completed regime; an all-time peak would
    # incorrectly label every later recovery date after a large historical crash.
    peak = df["equity_close"].rolling(window=60, min_periods=20).max()
    future_low = pd.concat([df["equity_close"].shift(-offset) for offset in range(lookahead_days + 1)], axis=1).min(axis=1)
    return (future_low.div(peak).sub(1.0) <= drawdown_threshold).astype(int).rename("crash_label")


if __name__ == "__main__":
    demo = Path(__file__).resolve().parents[1] / "data" / "prices.csv"
    prices = load_price_data(str(demo))
    print(prices.head(3))
    print(f"Crash-labelled rows: {label_crash_windows(prices).sum()}")
