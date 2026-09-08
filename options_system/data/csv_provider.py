"""Load OHLCV history from local CSV files.

Useful for offline backtesting against data you've exported from your broker,
TradingView, or any provider -- no network access required at all.

Expected file layout: <data_dir>/<TICKER>.csv with columns
date,open,high,low,close,volume (case-insensitive, extra columns ignored).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_system.data.base import PriceDataProvider, validate_ohlcv


class CsvProvider(PriceDataProvider):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        path = self.data_dir / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No CSV found for {ticker} at {path}. "
                f"Export daily OHLCV history to this path to use CsvProvider."
            )
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = "date" if "date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return validate_ohlcv(df, ticker)
