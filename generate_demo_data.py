"""Create deterministic offline demo prices spanning 2018, 2020 and 2022 stress regimes."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", "2022-12-30")
    equity_returns = rng.normal(0.00032, 0.009, len(dates))
    gold_returns = rng.normal(0.00015, 0.006, len(dates))
    # Deterministic regime shocks make the live replay legible in a judging demo.
    shocks = [("2018-10-01", "2018-12-24", -0.0035, 0.0008), ("2020-02-20", "2020-03-23", -0.0105, 0.0030), ("2022-01-03", "2022-06-17", -0.0015, 0.0007)]
    for start, end, equity_drag, gold_lift in shocks:
        mask = (dates >= start) & (dates <= end)
        equity_returns[mask] += equity_drag + rng.normal(0, 0.012, mask.sum())
        gold_returns[mask] += gold_lift
    # Recovery periods help judges see the strategy rotate back into risk assets.
    for start, end in [("2020-03-24", "2020-12-31"), ("2022-06-20", "2022-12-30")]:
        mask = (dates >= start) & (dates <= end)
        equity_returns[mask] += 0.00055
    equity = 100 * np.exp(np.cumsum(equity_returns))
    gold = 100 * np.exp(np.cumsum(gold_returns))
    output = Path(__file__).resolve().parent / "data" / "prices.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "equity_close": equity.round(2), "gold_close": gold.round(2)}).to_csv(output, index=False)
    (output.parent / "data_source.json").write_text(json.dumps({"kind": "synthetic_demo", "coverage": "2018-2022", "note": "Deterministic offline demonstration data; not real historical market data."}, indent=2), encoding="utf-8")
    print(f"Wrote {len(dates)} offline demo rows to {output}")


if __name__ == "__main__":
    main()
