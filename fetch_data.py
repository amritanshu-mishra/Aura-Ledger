"""Optional one-time historical fetcher. Never called by the dashboard at runtime."""
from pathlib import Path
import json


def main() -> None:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Optional fetcher requires: pip install yfinance") from exc
    equity = yf.download("^GSPC", start="2007-01-01", end="2024-01-01", auto_adjust=True, progress=False)["Close"]
    gold = yf.download("GLD", start="2007-01-01", end="2024-01-01", auto_adjust=True, progress=False)["Close"]
    import pandas as pd
    equity = equity.squeeze().rename("equity_close")
    gold = gold.squeeze().rename("gold_close")
    prices = pd.concat([equity, gold], axis=1).dropna().reset_index()
    prices = prices.rename(columns={"Date": "date"})
    output = Path(__file__).resolve().parent / "data" / "prices.csv"
    prices.to_csv(output, index=False)
    (output.parent / "data_source.json").write_text(json.dumps({"kind": "historical_market_data", "provider": "Yahoo Finance via yfinance", "assets": {"equity": "^GSPC", "gold": "GLD"}, "coverage": "2007-2023", "regimes": ["2008 Global Financial Crisis", "2018 Q4 selloff", "2020 COVID crash", "2022 drawdown"]}, indent=2), encoding="utf-8")
    print(f"Saved fetched prices to {output}")


if __name__ == "__main__":
    main()
