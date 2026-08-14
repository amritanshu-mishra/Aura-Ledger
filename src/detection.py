from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Return a trailing-window z-score, preserving early rows as NaN."""
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return series.sub(mean).div(std.replace(0, np.nan))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create early-warning signals from volatility, momentum, drawdown and hedge behaviour."""
    if "equity_close" not in df:
        raise ValueError("equity_close is required to compute risk features.")
    returns = df["equity_close"].pct_change()
    gold_returns = df["gold_close"].pct_change() if "gold_close" in df else pd.Series(index=df.index, dtype=float)
    rolling_volatility = returns.rolling(10, min_periods=10).std(ddof=0)
    drawdown = df["equity_close"].div(df["equity_close"].cummax()).sub(1.0)
    five_day_return = df["equity_close"].pct_change(5)
    relative_hedge_return = gold_returns.rolling(5, min_periods=5).sum().sub(returns.rolling(5, min_periods=5).sum())
    features = pd.DataFrame(index=df.index)
    features["volatility_z"] = rolling_zscore(rolling_volatility, window=20)
    features["drawdown_velocity_z"] = rolling_zscore(drawdown.diff(), window=20)
    features["momentum_stress_z"] = -rolling_zscore(five_day_return, window=20)
    features["hedge_divergence_z"] = rolling_zscore(relative_hedge_return, window=20)
    features["volatility_acceleration_z"] = rolling_zscore(rolling_volatility.diff(5), window=20)
    if "volume" in df.columns:
        features["liquidity_z"] = rolling_zscore(df["volume"].pct_change().abs(), window=20)
    return features.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    from pathlib import Path
    from src.data_loader import load_price_data
    prices = load_price_data(str(Path(__file__).resolve().parents[1] / "data" / "prices.csv"))
    print(compute_features(prices).dropna().head(3))
