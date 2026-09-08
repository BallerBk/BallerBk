"""Price data provider interface.

Every strategy and the backtester consume OHLCV data through this interface,
never through a hardcoded data source. That's what lets the same engine run
here (against synthetic data, since this sandbox has no market-data network
access) and on your own machine (against yfinance, a broker API, or a CSV
export) without touching strategy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PriceDataProvider(ABC):
    """Returns daily OHLCV bars for a ticker as a DataFrame.

    Required columns: open, high, low, close, volume
    Index: DatetimeIndex, ascending, one row per trading day.
    """

    @abstractmethod
    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_latest_price(self, ticker: str) -> float:
        """Convenience default: last close in the most recent available bar."""
        df = self.get_history_recent(ticker, lookback_days=5)
        if df.empty:
            raise ValueError(f"No price data available for {ticker}")
        return float(df["close"].iloc[-1])

    def get_history_recent(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=lookback_days * 2 + 10)  # pad for weekends/holidays
        df = self.get_history(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return df.tail(lookback_days)


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: data missing required columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: data index must be a DatetimeIndex")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
