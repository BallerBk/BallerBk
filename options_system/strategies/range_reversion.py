"""Range / Support-Resistance Mean Reversion.

Public/generic concept: in a range-bound market (flat EMA slope, i.e. no
established trend), fade moves to the outer Bollinger Band back toward the
mean. This is standard mean-reversion technical analysis. Structured as
credit spreads because the thesis is "price stays inside a boundary", which
is a premium-selling (theta-positive) thesis, not a directional bet.
"""

from __future__ import annotations

import pandas as pd

from options_system.indicators import bollinger_bands, ema
from options_system.strategies.base import BEARISH, BULLISH, NONE, Strategy


class RangeReversionStrategy(Strategy):
    key = "range_reversion"
    display_name = "Range / Support-Resistance Reversion"
    description = (
        "In range-bound conditions (flat EMA trend slope), sells premium "
        "against moves that tag the outer Bollinger Band, betting on "
        "reversion toward the mean."
    )
    default_structure = "credit_spread"

    def __init__(
        self,
        bb_len: int = 20,
        bb_std: float = 2.0,
        trend_filter_len: int = 50,
        flat_slope_threshold: float = 0.0015,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bb_len = bb_len
        self.bb_std = bb_std
        self.trend_filter_len = trend_filter_len
        self.flat_slope_threshold = flat_slope_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df.index)
        if len(df) < max(self.bb_len, self.trend_filter_len) + 10:
            return out

        close = df["close"]
        upper, mid, lower = bollinger_bands(close, self.bb_len, self.bb_std)
        trend_ema = ema(close, self.trend_filter_len)
        # normalized slope over 10 bars: near-zero => range-bound
        slope = (trend_ema - trend_ema.shift(10)) / trend_ema.shift(10)
        is_range = slope.abs() <= self.flat_slope_threshold

        touches_upper = (close >= upper) & is_range
        touches_lower = (close <= lower) & is_range

        band_width = (upper - lower) / mid
        # tighter ranges after a squeeze -> more conviction on the fade
        conviction = (1 - band_width.clip(0, 0.20) / 0.20).clip(0, 1)

        # Fade the upper band => expect reversion down => bearish credit spread (call side)
        out.loc[touches_upper, "direction"] = BEARISH
        out.loc[touches_upper, "confidence"] = (0.5 + 0.4 * conviction[touches_upper]).clip(0, 1)
        out.loc[touches_upper, "reason"] = "Range-bound market, price tagged upper Bollinger Band"
        out.loc[touches_upper, "stop_price"] = upper[touches_upper] * 1.01
        out.loc[touches_upper, "target_price"] = mid[touches_upper]

        # Fade the lower band => expect reversion up => bullish credit spread (put side)
        out.loc[touches_lower, "direction"] = BULLISH
        out.loc[touches_lower, "confidence"] = (0.5 + 0.4 * conviction[touches_lower]).clip(0, 1)
        out.loc[touches_lower, "reason"] = "Range-bound market, price tagged lower Bollinger Band"
        out.loc[touches_lower, "stop_price"] = lower[touches_lower] * 0.99
        out.loc[touches_lower, "target_price"] = mid[touches_lower]

        none_mask = ~(touches_upper | touches_lower)
        out.loc[none_mask, "direction"] = NONE
        return out
