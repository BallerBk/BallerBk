"""Opening Range Breakout (ORB).

Public/generic concept: define a range from the first N minutes of the
regular session, then trade the break of that range (confirmed by volume)
in the direction of the break, targeting an R-multiple of the range's
width, exiting same-day. This is one of the most widely published intraday
setups -- see e.g. Option Alpha's write-up on 0DTE ORB strategies.

Structurally different from the other three strategies: it needs intraday
(minute-level) bars, not daily bars, and the option side is same-day
(0DTE-style), not a 30-45 DTE position. It is NOT run through the daily-bar
backtest engine or the once-daily scanner -- see backtest/orb_engine.py.

0DTE options are high risk: time decay is extremely fast intraday, moves
against you can go to a near-total loss within minutes, and this strategy
produces at most one signal per ticker per day. Treat it as more aggressive
than the other three by design, not a bug.
"""

from __future__ import annotations

import pandas as pd

from options_system.strategies.base import BEARISH, BULLISH, Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    key = "orb"
    display_name = "Opening Range Breakout (0DTE)"
    description = (
        "Trades the break of the first N minutes' high/low range, "
        "confirmed by volume, with a same-day 0DTE-style option."
    )
    default_structure = "long_call"
    intraday = True

    def __init__(
        self,
        orb_minutes: int = 15,
        volume_mult: float = 1.2,
        target_r_mult: float = 1.5,
        session_open_hour: int = 9,
        session_open_minute: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orb_minutes = orb_minutes
        self.volume_mult = volume_mult
        self.target_r_mult = target_r_mult
        self.session_open_hour = session_open_hour
        self.session_open_minute = session_open_minute

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df.index)
        if df.empty:
            return out

        day_key = df.index.normalize()
        for day, day_df in df.groupby(day_key):
            day_open = day + pd.Timedelta(
                hours=self.session_open_hour, minutes=self.session_open_minute
            )
            orb_end = day_open + pd.Timedelta(minutes=self.orb_minutes)

            orb_bars = day_df[(day_df.index >= day_open) & (day_df.index < orb_end)]
            if orb_bars.empty:
                continue
            or_high = orb_bars["high"].max()
            or_low = orb_bars["low"].min()
            or_avg_vol = orb_bars["volume"].mean()
            if or_avg_vol <= 0 or or_high <= or_low:
                continue

            post_bars = day_df[day_df.index >= orb_end]
            if post_bars.empty:
                continue

            vol_confirmed = post_bars["volume"] >= self.volume_mult * or_avg_vol
            bullish_mask = (post_bars["close"] > or_high) & vol_confirmed
            bearish_mask = (post_bars["close"] < or_low) & vol_confirmed
            combined = bullish_mask | bearish_mask
            if not combined.any():
                continue

            first_ts = combined.idxmax()  # first True (only one signal per day)
            is_bullish = bool(bullish_mask.loc[first_ts])
            entry_price = float(post_bars.loc[first_ts, "close"])
            entry_vol = float(post_bars.loc[first_ts, "volume"])
            range_width = or_high - or_low
            vol_ratio = entry_vol / or_avg_vol
            vol_strength = min(max((vol_ratio - self.volume_mult) / self.volume_mult, 0), 1)
            confidence = min(max(0.55 + 0.45 * vol_strength, 0), 1)

            out.loc[first_ts, "direction"] = BULLISH if is_bullish else BEARISH
            out.loc[first_ts, "confidence"] = confidence
            if is_bullish:
                out.loc[first_ts, "reason"] = (
                    f"Broke above {self.orb_minutes}-min opening range high "
                    f"({or_high:.2f}), volume {vol_ratio:.1f}x opening-range avg"
                )
                out.loc[first_ts, "stop_price"] = or_low
                out.loc[first_ts, "target_price"] = entry_price + self.target_r_mult * range_width
            else:
                out.loc[first_ts, "reason"] = (
                    f"Broke below {self.orb_minutes}-min opening range low "
                    f"({or_low:.2f}), volume {vol_ratio:.1f}x opening-range avg"
                )
                out.loc[first_ts, "stop_price"] = or_high
                out.loc[first_ts, "target_price"] = entry_price - self.target_r_mult * range_width

        return out
