from __future__ import annotations

import hashlib
import json

import pandas as pd


class HashChainLedger:
    """Append-only decision log whose links reveal edited historical entries."""

    def __init__(self):
        self.entries = []

    @staticmethod
    def _hash_entry(index: int, timestamp: str, data: dict, previous_hash: str) -> str:
        payload = {"index": index, "timestamp": timestamp, "data": data, "previous_hash": previous_hash}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(previous_hash.encode("utf-8") + encoded).hexdigest()

    def log_decision(self, data: dict) -> str:
        timestamp = str(data.get("date", pd.Timestamp.now().isoformat()))
        previous_hash = self.entries[-1]["hash"] if self.entries else "GENESIS"
        index = len(self.entries)
        entry_hash = self._hash_entry(index, timestamp, data, previous_hash)
        self.entries.append({"index": index, "timestamp": timestamp, "data": data, "previous_hash": previous_hash, "hash": entry_hash})
        return entry_hash

    def verify_chain(self) -> bool:
        previous_hash = "GENESIS"
        for index, entry in enumerate(self.entries):
            if entry["index"] != index or entry["previous_hash"] != previous_hash:
                return False
            if entry["hash"] != self._hash_entry(index, entry["timestamp"], entry["data"], previous_hash):
                return False
            previous_hash = entry["hash"]
        return True

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for entry in reversed(self.entries):
            data = entry["data"]
            rows.append({"timestamp": entry["timestamp"], "action_taken": data.get("action_taken", {}).get("summary", "HOLD"), "crash_probability": round(float(data.get("crash_probability", 0.0)), 3), "signals": json.dumps(data.get("signals", {}), separators=(",", ":")), "hash": entry["hash"][:10]})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    ledger = HashChainLedger()
    ledger.log_decision({"date": "2020-03-10", "crash_probability": 0.81, "action_taken": {"summary": "DEFEND"}})
    print(ledger.to_dataframe())
    print(f"Chain valid: {ledger.verify_chain()}")
