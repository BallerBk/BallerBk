"""Live data via yfinance.

This is the provider you'll actually use once you run the system on a
machine with normal internet access (this sandbox blocks all market-data
domains, so it can't be exercised here). `yfinance` pulls free delayed data
from Yahoo Finance -- fine for daily-bar swing strategies, not for
professional-grade low-latency execution.

Install with: pip install yfinance
"""

from __future__ import annotations

import pandas as pd

from options_system.data.base import PriceDataProvider, validate_ohlcv


class YFinanceProvider(PriceDataProvider):
    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from e

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if raw.empty:
            raise ValueError(f"yfinance returned no data for {ticker} in [{start}, {end}]")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        return validate_ohlcv(raw[["open", "high", "low", "close", "volume"]], ticker)

    def get_intraday_history(
        self, ticker: str, start: str, end: str, bar_minutes: int = 5
    ) -> pd.DataFrame:
        """Intraday bars via yfinance. NOTE Yahoo's own limits, not ours:
        1-minute bars only cover the trailing ~7 days; 5/15/30-minute bars
        cover roughly the trailing ~60 days. Requesting a wider [start, end]
        than that will silently return less than you asked for -- yfinance
        doesn't error, it just truncates. Fine for validating/tuning ORB
        recently; not a source of years of intraday history.
        """
        import yfinance as yf

        interval = f"{bar_minutes}m"
        raw = yf.download(
            ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True
        )
        if raw.empty:
            raise ValueError(
                f"yfinance returned no intraday data for {ticker} in [{start}, {end}] "
                f"at {interval} -- note Yahoo's ~7/60 day lookback limits on intraday intervals."
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        return validate_ohlcv(raw[["open", "high", "low", "close", "volume"]], ticker)
