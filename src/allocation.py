from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

ASSETS = ("equity", "gold", "cash")


def lock_precrash_correlation(returns_df: pd.DataFrame, as_of_date, lookback_days: int = 60, buffer_days: int = 5) -> pd.DataFrame:
    """Freeze covariance using observations available before the action date."""
    clean = returns_df.rename(columns={"equity_close": "equity", "gold_close": "gold"})
    clean = clean[[c for c in ("equity", "gold") if c in clean]].dropna()
    eligible = clean.loc[clean.index <= pd.Timestamp(as_of_date)]
    if len(eligible) <= buffer_days + 5:
        raise ValueError("Not enough history to lock a pre-crash covariance snapshot.")
    covariance = eligible.iloc[:-buffer_days].tail(lookback_days).cov().reindex(index=["equity", "gold"], columns=["equity", "gold"], fill_value=0.0)
    full = pd.DataFrame(0.0, index=ASSETS, columns=ASSETS)
    full.loc[["equity", "gold"], ["equity", "gold"]] = covariance
    return full


def mvo_target_weights(expected_returns, cov_matrix, risk_aversion: float = 2.0) -> dict:
    """Solve long-only defensive MVO allocation with modest concentration caps."""
    expected = pd.Series(expected_returns, dtype=float).reindex(ASSETS).fillna(0.0).to_numpy()
    covariance = pd.DataFrame(cov_matrix, index=ASSETS, columns=ASSETS).to_numpy(dtype=float) + np.eye(3) * 1e-7
    weights = cp.Variable(3)
    problem = cp.Problem(cp.Maximize(expected @ weights - risk_aversion * cp.quad_form(weights, covariance)), [cp.sum(weights) == 1, weights >= 0, weights[0] <= 0.45, weights[1] <= 0.55, weights[2] >= 0.20])
    problem.solve(solver=cp.CLARABEL, verbose=False)
    if weights.value is None:
        return {"equity": 0.20, "gold": 0.50, "cash": 0.30}
    solved = np.maximum(np.asarray(weights.value).reshape(-1), 0)
    solved /= solved.sum()
    return {asset: float(weight) for asset, weight in zip(ASSETS, solved)}


def scale_allocation(base_weights: dict, defensive_weights: dict, crash_probability: float) -> dict:
    """Linearly blend baseline and defensive baskets as risk rises."""
    probability = float(np.clip(crash_probability, 0.0, 1.0))
    blended = {asset: float(base_weights.get(asset, 0) * (1 - probability) + defensive_weights.get(asset, 0) * probability) for asset in ASSETS}
    total = sum(blended.values())
    return {asset: weight / total for asset, weight in blended.items()}


if __name__ == "__main__":
    sample = pd.DataFrame({"equity": [0.01, -0.02, 0.005] * 30, "gold": [0.003, 0.004, -0.001] * 30}, index=pd.bdate_range("2020-01-01", periods=90))
    cov = lock_precrash_correlation(sample, sample.index[-1], buffer_days=1)
    print(mvo_target_weights({"equity": 0.03, "gold": 0.06, "cash": 0.02}, cov))
