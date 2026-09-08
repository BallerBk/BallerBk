"""IV-Rank Premium Selling.

Public/generic concept, closely following tastytrade's widely published
research: sell premium when volatility is elevated relative to its own
trailing range ("IV Rank" > 50), since elevated IV tends to mean-revert and
richly prices short options. Direction of the structure follows the broader
trend filter: credit spread with the trend if one exists, iron condor
(neutral, sell both sides) if the market is trendless.

NOTE: true IV Rank comes from the options chain's implied volatility, which
requires a paid/real-time options data feed. This strategy uses realized
(historical) volatility rank as a proxy -- see indicators.iv_rank_proxy for
the caveat and README.md for how to swap in a real IV data source.
"""

from __future__ import annotations

import pandas as pd

from options_system.indicators import atr, ema, iv_rank_proxy
from options_system.strategies.base import BEARISH, BULLISH, NEUTRAL_RANGE, NONE, Strategy


class IvRankPremiumStrategy(Strategy):
    key = "iv_rank_premium"
    display_name = "IV-Rank Premium Selling"
    description = (
        "Sells premium (credit spread or iron condor) when realized-vol "
        "rank (IV Rank proxy) exceeds 50, following tastytrade's published "
        "high-probability premium-selling research."
    )
    default_structure = "credit_spread"

    def __init__(
        self,
        vol_length: int = 20,
        rank_length: int = 252,
        iv_rank_threshold: float = 50.0,
        trend_len: int = 50,
        flat_slope_threshold: float = 0.0015,
        stop_atr_mult: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vol_length = vol_length
        self.rank_length = rank_length
        self.iv_rank_threshold = iv_rank_threshold
        self.trend_len = trend_len
        self.flat_slope_threshold = flat_slope_threshold
        self.stop_atr_mult = stop_atr_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df.index)
        if len(df) < max(self.rank_length // 4, self.trend_len) + 10:
            return out

        close = df["close"]
        ivr = iv_rank_proxy(close, self.vol_length, self.rank_length)
        trend_ema = ema(close, self.trend_len)
        slope = (trend_ema - trend_ema.shift(10)) / trend_ema.shift(10)
        atr_v = atr(df, 14)

        elevated = ivr >= self.iv_rank_threshold
        uptrend = elevated & (slope > self.flat_slope_threshold)
        downtrend = elevated & (slope < -self.flat_slope_threshold)
        flat = elevated & (slope.abs() <= self.flat_slope_threshold)

        conviction = ((ivr - self.iv_rank_threshold) / (100 - self.iv_rank_threshold)).clip(0, 1)

        # Uptrend + high IV rank -> sell put credit spread (bullish structure)
        out.loc[uptrend, "direction"] = BULLISH
        out.loc[uptrend, "confidence"] = (0.5 + 0.5 * conviction[uptrend]).clip(0, 1)
        out.loc[uptrend, "reason"] = "IV Rank proxy >= threshold in an uptrend: sell put credit spread"
        out.loc[uptrend, "stop_price"] = close[uptrend] - self.stop_atr_mult * atr_v[uptrend]
        out.loc[uptrend, "target_price"] = close[uptrend]  # target = premium decay, not price move

        # Downtrend + high IV rank -> sell call credit spread (bearish structure)
        out.loc[downtrend, "direction"] = BEARISH
        out.loc[downtrend, "confidence"] = (0.5 + 0.5 * conviction[downtrend]).clip(0, 1)
        out.loc[downtrend, "reason"] = "IV Rank proxy >= threshold in a downtrend: sell call credit spread"
        out.loc[downtrend, "stop_price"] = close[downtrend] + self.stop_atr_mult * atr_v[downtrend]
        out.loc[downtrend, "target_price"] = close[downtrend]

        # Flat + high IV rank -> iron condor (sell both sides)
        out.loc[flat, "direction"] = NEUTRAL_RANGE
        out.loc[flat, "confidence"] = (0.5 + 0.5 * conviction[flat]).clip(0, 1)
        out.loc[flat, "reason"] = "IV Rank proxy >= threshold, no trend: sell iron condor"
        out.loc[flat, "stop_price"] = close[flat] - self.stop_atr_mult * atr_v[flat]
        out.loc[flat, "target_price"] = close[flat] + self.stop_atr_mult * atr_v[flat]

        none_mask = ~(uptrend | downtrend | flat)
        out.loc[none_mask, "direction"] = NONE
        return out
