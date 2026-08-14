from __future__ import annotations

import numpy as np
import pandas as pd

from src.classifier import predict_crash_probability
from src.ledger import HashChainLedger
from src.portfolio import SimulatedBroker


DEFENSIVE_GOLD_SHARE = 0.85
DEFENSIVE_CASH_SHARE = 0.15


def _positive(value: float) -> float:
    """Turn a z-score into a bounded abnormality contribution."""
    return float(np.clip(value / 3.0, 0.0, 1.0))


def abnormal_market_score(feature_row: pd.Series, model_probability: float) -> float:
    """Calculate the transparent abnormal-market signal that controls trading.

    The trained classifier remains a minor input. Observable stress factors
    dominate the decision: volatility, drawdown speed, negative momentum,
    equity/gold divergence, and volatility acceleration.
    """
    market_stress = (
        0.22 * _positive(float(feature_row.get("volatility_z", 0.0)))
        + 0.25 * _positive(-float(feature_row.get("drawdown_velocity_z", 0.0)))
        + 0.28 * _positive(float(feature_row.get("momentum_stress_z", 0.0)))
        + 0.18 * _positive(float(feature_row.get("hedge_divergence_z", 0.0)))
        + 0.07 * _positive(float(feature_row.get("volatility_acceleration_z", 0.0)))
    )
    return float(np.clip(0.90 * market_stress + 0.10 * model_probability, 0.0, 1.0))


def _sell_all_then_buy_gold(broker: SimulatedBroker, prices: dict, date) -> list[dict]:
    """Sell the equity proxy, then fund a gold hedge from the proceeds."""
    orders = []
    equity_units = broker.get_holdings()["equity"]
    if equity_units > 1e-8:
        orders.append(broker.place_order("equity", equity_units, "sell", prices["equity"], date))
    gold_budget = broker.get_holdings()["cash"] * DEFENSIVE_GOLD_SHARE
    if gold_budget > 1e-8:
        orders.append(broker.place_order("gold", gold_budget / prices["gold"], "buy", prices["gold"], date))
    return orders


def _sell_gold_then_rebuy_equity(broker: SimulatedBroker, prices: dict, date) -> list[dict]:
    """Close the hedge and reinvest all capital in the original equity proxy."""
    orders = []
    gold_units = broker.get_holdings()["gold"]
    if gold_units > 1e-8:
        orders.append(broker.place_order("gold", gold_units, "sell", prices["gold"], date))
    equity_budget = broker.get_holdings()["cash"]
    if equity_budget > 1e-8:
        orders.append(broker.place_order("equity", equity_budget / prices["equity"], "buy", prices["equity"], date))
    return orders


def run_backtest(
    price_df: pd.DataFrame,
    features_df: pd.DataFrame,
    model,
    broker: SimulatedBroker,
    ledger: HashChainLedger,
    threshold: float = 0.65,
) -> pd.DataFrame:
    """Replay Aura Ledger's explicit sell-to-gold and recovery re-entry loop.

    The original portfolio starts 100% in the equity proxy. An abnormal signal
    triggers one all-equity sale, followed by an 85% gold / 15% cash defense.
    Once the market makes a low, risk normalises and price recovers by 8%, the
    gold position is sold and all capital repurchases that same equity proxy.
    Only state transitions trade, so normal replay days remain stable HOLDs.
    """
    baseline_cash = broker.get_holdings()["cash"]
    baseline_equity = broker.get_holdings()["equity"]
    rolling_peak = price_df["equity_close"].rolling(63, min_periods=1).max()
    mode = "normal"
    defensive_low: float | None = None
    defensive_days = 0
    rows = []

    for date, prices_row in price_df.iterrows():
        prices = {"equity": float(prices_row.equity_close), "gold": float(prices_row.gold_close)}
        equity_drawdown = float(prices["equity"] / rolling_peak.loc[date] - 1.0)
        feature_row = features_df.loc[date] if date in features_df.index else pd.Series(dtype=float)
        if feature_row.empty or feature_row.isna().any():
            model_probability = 0.0
            probability = 0.0
            signals = {name: None for name in features_df.columns}
        else:
            model_probability = float(predict_crash_probability(model, feature_row.to_numpy().reshape(1, -1))[0])
            probability = abnormal_market_score(feature_row, model_probability)
            signals = {name: round(float(value), 3) for name, value in feature_row.items()}

        orders: list[dict] = []
        action = "HOLD EQUITY"
        # A high score alone is not enough: price must also be meaningfully
        # below its recent peak. This filters benign one-day volatility spikes.
        if mode == "normal" and probability >= threshold and equity_drawdown <= -0.04:
            orders = _sell_all_then_buy_gold(broker, prices, date)
            mode, defensive_low, defensive_days = "defensive", prices["equity"], 0
            action = "SELL EQUITY -> BUY GOLD"
        elif mode == "defensive":
            defensive_days += 1
            defensive_low = min(float(defensive_low), prices["equity"])
            recovery_confirmed = (
                defensive_days >= 5
                and probability <= 0.25
                and float(feature_row.get("volatility_z", 0.0)) <= 0.0
                and prices["equity"] >= float(defensive_low) * 1.08
            )
            if recovery_confirmed:
                orders = _sell_gold_then_rebuy_equity(broker, prices, date)
                mode, defensive_low, defensive_days = "normal", None, 0
                action = "SELL GOLD -> REBUY EQUITY"
            else:
                action = "HOLD GOLD + CASH"

        action_payload = {
            "summary": action,
            "orders": orders,
            "execution_mode": mode,
            "model_probability": round(model_probability, 3),
            "abnormal_market_score": round(probability, 3),
            "equity_drawdown_from_63d_peak": round(equity_drawdown, 3),
        }
        ledger.log_decision({
            "date": str(date.date()),
            "crash_probability": probability,
            "signals": signals,
            "action_taken": action_payload,
            "mode": mode,
        })

        holdings = broker.get_holdings()
        value = broker.portfolio_value(prices)
        rows.append({
            "date": date,
            "strategy_value": value,
            "buyhold_value": baseline_cash + baseline_equity * prices["equity"],
            "crash_probability": probability,
            "allocation_equity_pct": holdings["equity"] * prices["equity"] / value,
            "allocation_gold_pct": holdings["gold"] * prices["gold"] / value,
            "allocation_cash_pct": holdings["cash"] / value,
            "action_taken": action,
            "mode": mode,
            "signals": signals,
        })
    return pd.DataFrame(rows)


def _series_metrics(values: pd.Series) -> dict:
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1)
    drawdown = values.div(values.cummax()).sub(1)
    max_drawdown = float(drawdown.min())
    downside = returns[returns < 0].std(ddof=0)
    sortino = float(returns.mean() / downside * np.sqrt(252)) if downside and not np.isnan(downside) else None
    years = max(len(values) / 252, 1 / 252)
    annualized = float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)
    calmar = float(annualized / abs(max_drawdown)) if max_drawdown < 0 else None
    return {"total_return": total_return, "max_drawdown": max_drawdown, "sortino_ratio": sortino, "calmar_ratio": calmar}


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Report return and risk metrics for strategy and buy-and-hold."""
    return {"strategy": _series_metrics(results_df["strategy_value"]), "buy_and_hold": _series_metrics(results_df["buyhold_value"])}


HISTORICAL_REGIMES = {
    "2008 Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "2018 Q4 selloff": ("2018-09-20", "2018-12-24"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 inflation drawdown": ("2022-01-03", "2022-10-12"),
}


def compute_regime_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create an event-level audit table for the four stress regimes in the pitch."""
    rows = []
    for regime, (start, end) in HISTORICAL_REGIMES.items():
        window = results_df.loc[(results_df["date"] >= start) & (results_df["date"] <= end)].copy()
        if len(window) < 2:
            continue
        strategy_return = window["strategy_value"].iloc[-1] / window["strategy_value"].iloc[0] - 1
        unhedged_return = window["buyhold_value"].iloc[-1] / window["buyhold_value"].iloc[0] - 1
        strategy_dd = window["strategy_value"].div(window["strategy_value"].cummax()).sub(1).min()
        unhedged_dd = window["buyhold_value"].div(window["buyhold_value"].cummax()).sub(1).min()
        rows.append({
            "regime": regime,
            "period": f"{window['date'].iloc[0]:%d %b %Y} - {window['date'].iloc[-1]:%d %b %Y}",
            "hedged_return": strategy_return,
            "unhedged_return": unhedged_return,
            "hedged_max_drawdown": strategy_dd,
            "unhedged_max_drawdown": unhedged_dd,
            "drawdown_protection": unhedged_dd - strategy_dd,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    from src.classifier import load_model
    from src.data_loader import load_price_data
    from src.detection import compute_features

    root = Path(__file__).resolve().parents[1]
    prices = load_price_data(str(root / "data" / "prices.csv"))
    results = run_backtest(prices, compute_features(prices), load_model(root / "models" / "crash_classifier.pkl"), SimulatedBroker(), HashChainLedger())
    print(compute_metrics(results))
