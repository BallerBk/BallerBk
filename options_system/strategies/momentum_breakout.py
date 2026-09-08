"""Momentum Breakout.

Public/generic concept: enter in the direction of a new N-day high/low
(Donchian channel breakout) confirmed by above-average volume, with an
ATR-based stop and an R-multiple target. This mechanical volume+breakout
formulation is standard, widely published breakout-trading methodology, not
specific to any one educator.

Trades directionally with long calls/puts (debit), since the thesis is a
sharp directional move where you want convexity, not premium collection.
"""

from __future__ import annotations

import pandas as pd

from options_system.indicators import atr, donchian, volume_ratio
from options_system.strategies.base import BEARISH, BULLISH, NONE, Strategy


class MomentumBreakoutStrategy(Strategy):
    key = "momentum_breakout"
    display_name = "Momentum Breakout"
    description = (
        "Buys breakouts above an N-day high (or below an N-day low) when "
        "confirmed by volume >= 1.2x its 20-bar average."
    )
    default_structure = "long_call"

    def __init__(
        self,
        channel_len: int = 20,
        volume_len: int = 20,
        volume_mult: float = 1.2,
        stop_atr_mult: float = 1.5,
        target_r_mult: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channel_len = channel_len
        self.volume_len = volume_len
        self.volume_mult = volume_mult
        self.stop_atr_mult = stop_atr_mult
        self.target_r_mult = target_r_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df.index)
        if len(df) < self.channel_len + 5:
            return out

        close = df["close"]
        upper, lower = donchian(df, self.channel_len)
        # Compare to *prior* channel so the breakout bar's own high/low doesn't
        # trivially satisfy "new high" against itself.
        prior_upper = upper.shift(1)
        prior_lower = lower.shift(1)
        vol_ratio = volume_ratio(df["volume"], self.volume_len)
        atr_v = atr(df, 14)

        vol_confirmed = vol_ratio >= self.volume_mult
        breakout_up = (close > prior_upper) & vol_confirmed
        breakout_down = (close < prior_lower) & vol_confirmed

        vol_strength = ((vol_ratio - self.volume_mult) / self.volume_mult).clip(0, 1)

        out.loc[breakout_up, "direction"] = BULLISH
        out.loc[breakout_up, "confidence"] = (0.55 + 0.45 * vol_strength[breakout_up]).clip(0, 1)
        out.loc[breakout_up, "reason"] = (
            f"New {self.channel_len}-day high, volume >= {self.volume_mult}x avg"
        )
        out.loc[breakout_up, "stop_price"] = close[breakout_up] - self.stop_atr_mult * atr_v[breakout_up]
        out.loc[breakout_up, "target_price"] = close[breakout_up] + (
            self.stop_atr_mult * self.target_r_mult * atr_v[breakout_up]
        )

        out.loc[breakout_down, "direction"] = BEARISH
        out.loc[breakout_down, "confidence"] = (
            0.55 + 0.45 * vol_strength[breakout_down]
        ).clip(0, 1)
        out.loc[breakout_down, "reason"] = (
            f"New {self.channel_len}-day low, volume >= {self.volume_mult}x avg"
        )
        out.loc[breakout_down, "stop_price"] = close[breakout_down] + self.stop_atr_mult * atr_v[breakout_down]
        out.loc[breakout_down, "target_price"] = close[breakout_down] - (
            self.stop_atr_mult * self.target_r_mult * atr_v[breakout_down]
        )

        none_mask = ~(breakout_up | breakout_down)
        out.loc[none_mask, "direction"] = NONE
        return out
