import unittest

import pandas as pd

from src.allocation import scale_allocation
from src.backtest import compute_regime_metrics
from src.data_loader import label_crash_windows
from src.ledger import HashChainLedger


class AuraLedgerCoreTests(unittest.TestCase):
    def test_probability_scaled_allocation_is_long_only_and_sums_to_one(self):
        weights = scale_allocation({"equity": 0.8, "gold": 0.1, "cash": 0.1}, {"equity": 0.2, "gold": 0.5, "cash": 0.3}, 0.6)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(value >= 0 for value in weights.values()))

    def test_hash_chain_detects_tampering(self):
        ledger = HashChainLedger()
        ledger.log_decision({"date": "2020-03-10", "crash_probability": 0.8, "action_taken": {"summary": "REBALANCE"}})
        self.assertTrue(ledger.verify_chain())
        ledger.entries[0]["data"]["crash_probability"] = 0.1
        self.assertFalse(ledger.verify_chain())

    def test_crash_labels_align_to_prices(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame({"equity_close": [100] * 50 + [85] * 50}, index=dates)
        labels = label_crash_windows(prices, lookahead_days=30)
        self.assertEqual(len(labels), len(prices))
        self.assertGreater(labels.sum(), 0)

    def test_regime_table_has_comparison_columns(self):
        dates = pd.bdate_range("2007-10-09", "2009-03-09")
        results = pd.DataFrame({"date": dates, "strategy_value": range(100, 100 + len(dates)), "buyhold_value": range(100, 100 + len(dates))})
        table = compute_regime_metrics(results)
        self.assertIn("drawdown_protection", table.columns)


if __name__ == "__main__":
    unittest.main()
