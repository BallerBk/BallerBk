"""Trend-Pullback Continuation.

Public/generic concept: in an established trend (fast EMA above/below slow
EMA), wait for price to pull back toward the fast EMA rather than chasing
strength, and confirm the pullback hasn't turned into a reversal via RSI.
This is one of the most widely taught trend-following entry patterns
(a variant appears throughout technical-analysis literature under names like
"buy the dip in an uptrend" / "EMA pullback").

Trades directionally with long calls/puts (debit) since the thesis is a
continued directional move, not a volatility crush.
"""

from __future__ import annotations

import pandas as pd

from options_system.indicators import atr, ema, rsi
from options_system.strategies.base import BEARISH, BULLISH, NONE, Strategy


class TrendPullbackStrategy(Strategy):
    key = "trend_pullback"
    display_name = "Trend Pullback Continuation"
    description = (
        "Buys pullbacks to the fast EMA within an established EMA trend, "
        "filtered by RSI to avoid catching a reversal."
    )
    default_structure = "long_call"

    def __init__(
        self,
        fast_len: int = 21,
        slow_len: int = 50,
        rsi_len: int = 14,
        rsi_low: float = 40,
        rsi_high: float = 60,
        pullback_atr_mult: float = 1.0,
        stop_atr_mult: float = 1.5,
        target_r_mult: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fast_len = fast_len
        self.slow_len = slow_len
        self.rsi_len = rsi_len
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.pullback_atr_mult = pullback_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.target_r_mult = target_r_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df.index)
        if len(df) < self.slow_len + 5:
            return out

        close = df["close"]
        fast = ema(close, self.fast_len)
        slow = ema(close, self.slow_len)
        rsi_v = rsi(close, self.rsi_len)
        atr_v = atr(df, 14)

        dist_to_fast = (close - fast).abs()
        near_fast = dist_to_fast <= (atr_v * self.pullback_atr_mult)
        rsi_neutral = (rsi_v >= self.rsi_low) & (rsi_v <= self.rsi_high)

        uptrend = fast > slow
        downtrend = fast < slow

        bullish = uptrend & near_fast & rsi_neutral & (close >= fast)
        bearish = downtrend & near_fast & rsi_neutral & (close <= fast)

        trend_strength = ((fast - slow).abs() / slow).clip(0, 0.10) / 0.10  # 0..1

        out.loc[bullish, "direction"] = BULLISH
        out.loc[bullish, "confidence"] = (0.5 + 0.5 * trend_strength[bullish]).clip(0, 1)
        out.loc[bullish, "reason"] = "Uptrend (EMA21>EMA50), pullback to EMA21, RSI neutral"
        out.loc[bullish, "stop_price"] = close[bullish] - self.stop_atr_mult * atr_v[bullish]
        out.loc[bullish, "target_price"] = close[bullish] + (
            self.stop_atr_mult * self.target_r_mult * atr_v[bullish]
        )

        out.loc[bearish, "direction"] = BEARISH
        out.loc[bearish, "confidence"] = (0.5 + 0.5 * trend_strength[bearish]).clip(0, 1)
        out.loc[bearish, "reason"] = "Downtrend (EMA21<EMA50), pullback to EMA21, RSI neutral"
        out.loc[bearish, "stop_price"] = close[bearish] + self.stop_atr_mult * atr_v[bearish]
        out.loc[bearish, "target_price"] = close[bearish] - (
            self.stop_atr_mult * self.target_r_mult * atr_v[bearish]
        )

        none_mask = ~(bullish | bearish)
        out.loc[none_mask, "direction"] = NONE
        return out
