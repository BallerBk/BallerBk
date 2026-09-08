"""Event-driven backtester.

Walks each ticker's price history bar by bar. When a strategy has no open
position and its signal fires, opens a simulated option trade (priced via
Black-Scholes through ContractSelector). While a position is open, reprices
it daily and applies structure-appropriate exit rules:

  - Directional (long_call/long_put): exit at the signal's stop/target price,
    or at expiration.
  - Premium-selling (credit_spread/iron_condor): exit at 50% of max profit,
    or at 21 DTE, or at expiration -- tastytrade's published management
    rules (see README/strategy docstrings for the source).

Approximation note: volatility is held constant at the entry-day estimate
for the life of each simulated trade (no forward vol path is available from
price history alone). This is a simplification, not a claim that real IV
stays flat -- see README "Backtest limitations".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from options_system.indicators import realized_vol
from options_system.options.black_scholes import price_and_greeks
from options_system.options.contract_selector import ContractSelector
from options_system.strategies.base import BEARISH, BULLISH, NONE, Strategy

DIRECTIONAL_STRUCTURES = {"long_call", "long_put"}
PREMIUM_STRUCTURES = {"put_credit_spread", "call_credit_spread", "iron_condor"}
TAKE_PROFIT_FRACTION = 0.50  # tastytrade convention: close at 50% of max credit
MANAGE_AT_DTE = 21  # tastytrade convention: manage/close at 21 DTE


@dataclass
class Trade:
    ticker: str
    strategy: str
    structure: str
    direction: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None = None
    entry_underlying: float = 0.0
    exit_underlying: float | None = None
    dte_entry: int = 0
    entry_sigma: float = 0.0
    max_profit: float | None = None
    max_loss: float | None = None
    pnl: float | None = None
    exit_reason: str | None = None
    confidence: float = 0.0
    legs: list = field(default_factory=list)


def _reprice_leg(leg: dict, s: float, t: float, r: float, sigma: float) -> float:
    if t <= 0:
        return max(0.0, (s - leg["strike"]) if leg["type"] == "call" else (leg["strike"] - s))
    return price_and_greeks(s, leg["strike"], t, r, sigma, leg["type"]).price


def _leg_cash_flows(legs: list[dict], current_prices: list[float]) -> float:
    """Net cash flow (per-share) of closing all legs at current_prices, mirroring
    the entry convention: sell leg entry credit = +price, buy leg entry debit = -price.
    """
    total = 0.0
    for leg, cur in zip(legs, current_prices):
        total += -cur if leg["side"] == "sell" else cur
    return total


def _entry_cash_flow(legs: list[dict]) -> float:
    total = 0.0
    for leg in legs:
        total += leg["est_price"] if leg["side"] == "sell" else -leg["est_price"]
    return total


def backtest_strategy(
    ticker: str,
    df: pd.DataFrame,
    strategy: Strategy,
    selector: ContractSelector,
    vol_length: int = 20,
    risk_free_rate: float = 0.045,
) -> list[Trade]:
    signals = strategy.generate_signals(df)
    sigma_series = realized_vol(df["close"], length=vol_length).clip(lower=0.05)

    trades: list[Trade] = []
    open_trade: Trade | None = None
    entry_idx: int | None = None

    dates = df.index
    for i, date in enumerate(dates):
        price = float(df["close"].iloc[i])
        sigma = float(sigma_series.iloc[i]) if pd.notna(sigma_series.iloc[i]) else 0.20

        if open_trade is not None:
            days_elapsed = (date - open_trade.entry_date).days
            days_remaining = open_trade.dte_entry - days_elapsed
            t = max(days_remaining, 0) / 365

            if open_trade.structure in DIRECTIONAL_STRUCTURES:
                sig_row = signals.iloc[entry_idx]
                stop, target = sig_row["stop_price"], sig_row["target_price"]
                hit_stop = pd.notna(stop) and (
                    (open_trade.direction == BULLISH and price <= stop)
                    or (open_trade.direction == BEARISH and price >= stop)
                )
                hit_target = pd.notna(target) and (
                    (open_trade.direction == BULLISH and price >= target)
                    or (open_trade.direction == BEARISH and price <= target)
                )
                expired = days_remaining <= 0
                if hit_stop or hit_target or expired:
                    leg = open_trade.legs[0]
                    cur_price = _reprice_leg(leg, price, t, risk_free_rate, open_trade.entry_sigma)
                    pnl_per_share = _entry_cash_flow(open_trade.legs) + cur_price
                    open_trade.exit_date = date
                    open_trade.exit_underlying = price
                    open_trade.pnl = round(pnl_per_share * 100, 2)
                    open_trade.exit_reason = "target" if hit_target else ("stop" if hit_stop else "expired")
                    trades.append(open_trade)
                    open_trade, entry_idx = None, None

            elif open_trade.structure in PREMIUM_STRUCTURES:
                cur_prices = [
                    _reprice_leg(leg, price, t, risk_free_rate, open_trade.entry_sigma)
                    for leg in open_trade.legs
                ]
                unrealized_per_share = _entry_cash_flow(open_trade.legs) + _leg_cash_flows(
                    open_trade.legs, cur_prices
                )
                unrealized = unrealized_per_share * 100
                profit_fraction = (
                    unrealized / open_trade.max_profit if open_trade.max_profit else 0.0
                )
                expired = days_remaining <= 0
                take_profit = profit_fraction >= TAKE_PROFIT_FRACTION
                manage_exit = days_remaining <= MANAGE_AT_DTE and not take_profit

                if take_profit or manage_exit or expired:
                    open_trade.exit_date = date
                    open_trade.exit_underlying = price
                    open_trade.pnl = round(unrealized, 2)
                    open_trade.exit_reason = (
                        "take_profit_50pct" if take_profit else ("manage_21dte" if manage_exit else "expired")
                    )
                    trades.append(open_trade)
                    open_trade, entry_idx = None, None

        if open_trade is None:
            direction = signals["direction"].iloc[i]
            if direction != NONE and i < len(dates) - 1:  # need room to hold the trade
                idea = selector.build(price, sigma, strategy.default_structure, direction)
                open_trade = Trade(
                    ticker=ticker,
                    strategy=strategy.key,
                    structure=idea.structure,
                    direction=direction,
                    entry_date=date,
                    entry_underlying=price,
                    dte_entry=idea.dte,
                    entry_sigma=sigma,
                    max_profit=idea.max_profit,
                    max_loss=idea.max_loss,
                    confidence=float(signals["confidence"].iloc[i]),
                    legs=idea.legs,
                )
                entry_idx = i

    return trades


def backtest_universe(
    tickers: list[str],
    data_provider,
    strategies: list[Strategy],
    start: str,
    end: str,
    selector: ContractSelector | None = None,
) -> pd.DataFrame:
    selector = selector or ContractSelector()
    all_trades: list[Trade] = []
    for ticker in tickers:
        df = data_provider.get_history(ticker, start, end)
        if df.empty:
            continue
        for strategy in strategies:
            trades = backtest_strategy(ticker, df, strategy, selector)
            all_trades.extend(trades)

    if not all_trades:
        return pd.DataFrame(
            columns=[
                "ticker", "strategy", "structure", "direction", "entry_date", "exit_date",
                "entry_underlying", "exit_underlying", "dte_entry", "pnl", "exit_reason", "confidence",
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
                "dte_entry": t.dte_entry,
                "max_profit": t.max_profit,
                "max_loss": t.max_loss,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "confidence": t.confidence,
            }
            for t in all_trades
        ]
    )
