"""Loads config.yaml into a plain dict. Kept intentionally simple -- no schema
library dependency, just sane defaults merged with whatever the user overrides.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "universe": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD"],
    "strategies": ["trend_pullback", "momentum_breakout", "range_reversion", "iv_rank_premium"],
    "account_size": 10000,
    "risk_pct_per_trade": 2.0,
    "scan_cadence": "once_daily_after_close",
    "backtest_lookback_years": 3,
    "data_provider": "synthetic",  # "synthetic" | "yfinance" | "csv"
    "csv_data_dir": "data",
    "risk_free_rate": 0.045,
}


def load_config(path: str = "config.yaml") -> dict:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        with p.open() as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg
