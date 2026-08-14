from __future__ import annotations

from copy import deepcopy

import pandas as pd


class SimulatedBroker:
    """Broker-shaped, fully in-memory execution adapter for the prototype."""

    def __init__(self, starting_cash: float = 0.0, starting_equity_units: float = 100.0):
        self.holdings = {"cash": float(starting_cash), "equity": float(starting_equity_units), "gold": 0.0}

    def get_holdings(self) -> dict:
        return deepcopy(self.holdings)

    def get_quote(self, asset: str, price_df: pd.DataFrame, current_date) -> float:
        if asset == "cash":
            return 1.0
        column = f"{asset}_close"
        if column not in price_df:
            raise KeyError(f"No price column found for {asset}.")
        return float(price_df.loc[:pd.Timestamp(current_date), column].iloc[-1])

    def place_order(self, asset: str, quantity: float, side: str, price: float, timestamp=None) -> dict:
        if asset not in {"equity", "gold"} or side not in {"buy", "sell"}:
            raise ValueError("Only equity/gold buy or sell orders are supported.")
        quantity = float(max(quantity, 0.0))
        if side == "sell":
            quantity = min(quantity, self.holdings[asset])
            self.holdings[asset] -= quantity
            self.holdings["cash"] += quantity * float(price)
        else:
            quantity = min(quantity, self.holdings["cash"] / float(price))
            self.holdings[asset] += quantity
            self.holdings["cash"] -= quantity * float(price)
        return {"timestamp": str(timestamp) if timestamp is not None else pd.Timestamp.now().isoformat(), "asset": asset, "side": side, "quantity": round(quantity, 6), "price": round(float(price), 4), "resulting_holdings": self.get_holdings()}

    def get_positions(self) -> dict:
        return {asset: self.holdings[asset] for asset in ("equity", "gold")}

    def portfolio_value(self, prices: dict) -> float:
        return float(self.holdings["cash"] + self.holdings["equity"] * prices["equity"] + self.holdings["gold"] * prices["gold"])


if __name__ == "__main__":
    broker = SimulatedBroker()
    print(broker.place_order("equity", 10, "sell", 100, "2020-01-01"))
