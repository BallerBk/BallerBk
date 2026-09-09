"""Dedicated backtest loop for Opening Range Breakout (0DTE, intraday).

Kept separate from backtest/engine.py deliberately: ORB trades open and
close within the same session using minute bars, priced against
minutes-remaining-until-close rather than calendar DTE, which is different
enough machinery that folding it into the daily-bar engine would make both
harder to read. Reuses the same Black-Scholes pricing model, so results are
mixed alongside the other four strategies' summaries via the same
metrics.summarize()/rank_strategies() functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from options_system.options.black_scholes import price_and_greeks, round_to_strike_increment, strike_for_target_delta
from options_system.options.contract_selector import default_strike_increment
from options_system.strategies.base import BEARISH, NONE
from options_system.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy

TRADING_DAYS_PER_YEAR = 252
SESSION_MINUTES = 390  # 9:30 - 16:00 ET regular session


@dataclass
class OrbTrade:
    ticker: str
    strategy: str = "orb"
    structure: str = "long_call"
    direction: str = ""
    entry_date: pd.Timestamp = None  # entry timestamp (intraday)
    exit_date: pd.Timestamp = None
    entry_underlying: float = 0.0
    exit_underlying: float | None = None
    strike: float = 0.0
    option_type: str = "call"
    entry_option_price: float = 0.0
    exit_option_price: float | None = None
    entry_sigma: float = 0.0
    pnl: float | None = None
    max_loss: float | None = None
    exit_reason: str | None = None
    confidence: float = 0.0


def _infer_bar_minutes(df: pd.DataFrame) -> int:
    day_key = df.index.normalize()
    first_day = day_key.unique()[0]
    day_df = df[day_key == first_day]
    if len(day_df) < 2:
        return 5
    diffs = day_df.index.to_series().diff().dropna()
    return max(1, int(diffs.median().total_seconds() // 60))


def _intraday_rolling_sigma(df: pd.DataFrame, bar_minutes: int, lookback_days: int = 5) -> pd.Series:
    """Annualized volatility estimated from intraday bar-to-bar returns,
    excluding the overnight jump between each day's last bar and the next
    day's first bar (that gap isn't part of the intraday distribution)."""
    day_key = df.index.normalize()
    log_ret = df.groupby(day_key)["close"].transform(lambda s: np.log(s / s.shift(1)))
    bars_per_day = max(1, SESSION_MINUTES // bar_minutes)
    bars_per_year = TRADING_DAYS_PER_YEAR * bars_per_day
    lookback_bars = max(bars_per_day, bars_per_day * lookback_days)
    rolling_std = log_ret.rolling(lookback_bars, min_periods=max(10, lookback_bars // 3)).std(ddof=0)
    return (rolling_std * np.sqrt(bars_per_year)).clip(lower=0.05)


def backtest_orb_strategy(
    ticker: str,
    df: pd.DataFrame,
    strategy: OpeningRangeBreakoutStrategy | None = None,
    target_delta: float = 0.50,
    risk_free_rate: float = 0.045,
    session_close_hour: int = 16,
    session_close_minute: int = 0,
) -> list[OrbTrade]:
    strategy = strategy or OpeningRangeBreakoutStrategy()
    if df.empty:
        return []

    bar_minutes = _infer_bar_minutes(df)
    sigma_series = _intraday_rolling_sigma(df, bar_minutes)
    signals = strategy.generate_signals(df)

    trades: list[OrbTrade] = []
    day_key = df.index.normalize()

    for day, day_df in df.groupby(day_key):
        day_signals = signals.loc[day_df.index]
        fired = day_signals[day_signals["direction"] != NONE]
        if fired.empty:
            continue
        entry_ts = fired.index[0]
        sig_row = fired.iloc[0]
        direction = sig_row["direction"]
        option_type = "put" if direction == BEARISH else "call"

        entry_price = float(day_df.loc[entry_ts, "close"])
        sigma = float(sigma_series.loc[entry_ts]) if pd.notna(sigma_series.loc[entry_ts]) else 0.30

        session_close = day + pd.Timedelta(hours=session_close_hour, minutes=session_close_minute)
        minutes_remaining = (session_close - entry_ts).total_seconds() / 60
        if minutes_remaining <= 0:
            continue
        t = minutes_remaining / (TRADING_DAYS_PER_YEAR * SESSION_MINUTES)

        increment = default_strike_increment(entry_price)
        raw_strike = strike_for_target_delta(entry_price, t, risk_free_rate, sigma, option_type, target_delta)
        strike = round_to_strike_increment(raw_strike, increment)
        entry_greeks = price_and_greeks(entry_price, strike, t, risk_free_rate, sigma, option_type)
        entry_option_price = entry_greeks.price
        if entry_option_price <= 0:
            continue

        remaining_bars = day_df.loc[day_df.index > entry_ts]
        exit_ts, exit_price_underlying, exit_reason = None, None, None
        for ts, row in remaining_bars.iterrows():
            price = float(row["close"])
            hit_stop = (
                (direction != BEARISH and price <= sig_row["stop_price"])
                or (direction == BEARISH and price >= sig_row["stop_price"])
            )
            hit_target = (
                (direction != BEARISH and price >= sig_row["target_price"])
                or (direction == BEARISH and price <= sig_row["target_price"])
            )
            if hit_stop or hit_target:
                exit_ts, exit_price_underlying = ts, price
                exit_reason = "target" if hit_target else "stop"
                break
        if exit_ts is None:
            exit_ts = day_df.index[-1]
            exit_price_underlying = float(day_df.loc[exit_ts, "close"])
            exit_reason = "session_close"

        minutes_remaining_exit = max((session_close - exit_ts).total_seconds() / 60, 0)
        t_exit = minutes_remaining_exit / (TRADING_DAYS_PER_YEAR * SESSION_MINUTES)
        exit_greeks = price_and_greeks(exit_price_underlying, strike, t_exit, risk_free_rate, sigma, option_type)
        exit_option_price = exit_greeks.price

        pnl = round((exit_option_price - entry_option_price) * 100, 2)

        trades.append(
            OrbTrade(
                ticker=ticker,
                structure=f"long_{option_type}",
                direction=direction,
                entry_date=entry_ts,
                exit_date=exit_ts,
                entry_underlying=entry_price,
                exit_underlying=exit_price_underlying,
                strike=strike,
                option_type=option_type,
                entry_option_price=round(entry_option_price, 2),
                exit_option_price=round(exit_option_price, 2),
                entry_sigma=sigma,
                pnl=pnl,
                max_loss=round(entry_option_price * 100, 2),
                exit_reason=exit_reason,
                confidence=float(sig_row["confidence"]),
            )
        )

    return trades


def backtest_orb_universe(
    tickers: list[str],
    data_provider,
    strategy: OpeningRangeBreakoutStrategy | None = None,
    start: str = None,
    end: str = None,
    bar_minutes: int = 5,
    **kwargs,
) -> pd.DataFrame:
    strategy = strategy or OpeningRangeBreakoutStrategy()
    all_trades: list[OrbTrade] = []
    for ticker in tickers:
        try:
            df = data_provider.get_intraday_history(ticker, start, end, bar_minutes=bar_minutes)
        except NotImplementedError as e:
            print(f"[orb-backtest] {e}")
            continue
        if df.empty:
            continue
        all_trades.extend(backtest_orb_strategy(ticker, df, strategy, **kwargs))

    if not all_trades:
        return pd.DataFrame(
            columns=[
                "ticker", "strategy", "structure", "direction", "entry_date", "exit_date",
                "entry_underlying", "exit_underlying", "strike", "option_type", "pnl",
                "exit_reason", "confidence",
            ]
        )

    return pd.DataFrame(
        [
            {
                "ticker": t.ticker,
                "strategy": t.strategy,
                "structure": t.structure,
                "direction": t.direction,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_underlying": t.entry_underlying,
                "exit_underlying": t.exit_underlying,
                "strike": t.strike,
                "option_type": t.option_type,
                "max_loss": t.max_loss,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "confidence": t.confidence,
            }
            for t in all_trades
        ]
    )
