"""Shared technical indicators used by every strategy module.

Pure functions over a price DataFrame (open/high/low/close/volume). Kept
separate from strategy logic so the Pine Script versions can reference the
exact same definitions in their comments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, length: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def donchian(df: pd.DataFrame, length: int = 20) -> tuple[pd.Series, pd.Series]:
    upper = df["high"].rolling(length).max()
    lower = df["low"].rolling(length).min()
    return upper, lower


def realized_vol(series: pd.Series, length: int = 20, annualize: bool = True) -> pd.Series:
    log_ret = np.log(series / series.shift(1))
    vol = log_ret.rolling(length).std(ddof=0)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def iv_rank_proxy(series: pd.Series, vol_length: int = 20, rank_length: int = 252) -> pd.Series:
    """Percentile rank (0-100) of current realized vol within its trailing window.

    This is an approximation of tastytrade-style "IV Rank" -- a real IV rank
    uses the option chain's implied volatility, which isn't freely available
    historically. Realized (historical) volatility rank is the standard
    academic/retail proxy when only price history is on hand. Swap this out
    for true IV rank if you wire in a data source that provides it (see
    README: "Improving the IV-rank proxy").
    """
    rv = realized_vol(series, length=vol_length)
    return rv.rolling(rank_length, min_periods=max(20, rank_length // 4)).rank(pct=True) * 100


def volume_ratio(volume: pd.Series, length: int = 20) -> pd.Series:
    avg = volume.rolling(length).mean()
    return volume / avg.replace(0, np.nan)
