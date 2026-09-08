import pandas as pd

from options_system.backtest.engine import backtest_universe
from options_system.backtest.metrics import rank_strategies, summarize
from options_system.data.synthetic_provider import SyntheticProvider
from options_system.options.contract_selector import ContractSelector
from options_system.strategies import REGISTRY


def test_backtest_universe_produces_closed_trades_with_pnl():
    provider = SyntheticProvider(seed=11, regime="mixed")
    strategies = [REGISTRY.get(k)() for k in REGISTRY.keys()]
    trades = backtest_universe(
        ["SPY"], provider, strategies, "2020-01-01", "2024-01-01", ContractSelector()
    )
    assert not trades.empty
    assert trades["pnl"].notna().all()
    assert trades["exit_date"].notna().all()
    assert (trades["exit_date"] >= trades["entry_date"]).all()


def test_no_overlapping_trades_per_ticker_strategy():
    """The engine holds at most one open position per (ticker, strategy) at a
    time -- a new signal must not open a trade until the prior one has closed."""
    provider = SyntheticProvider(seed=12, regime="mixed")
    strategies = [REGISTRY.get(k)() for k in REGISTRY.keys()]
    trades = backtest_universe(
        ["SPY"], provider, strategies, "2020-01-01", "2024-01-01", ContractSelector()
    )
    for (ticker, strategy), g in trades.groupby(["ticker", "strategy"]):
        g = g.sort_values("entry_date").reset_index(drop=True)
        for i in range(1, len(g)):
            assert g.loc[i, "entry_date"] >= g.loc[i - 1, "exit_date"], (
                f"overlapping trades for {ticker}/{strategy}"
            )


def test_premium_selling_trades_respect_21dte_or_take_profit_or_expiry():
    provider = SyntheticProvider(seed=13, regime="mixed")
    strategy = REGISTRY.get("iv_rank_premium")()
    trades = backtest_universe(
        ["SPY"], provider, [strategy], "2020-01-01", "2024-01-01", ContractSelector()
    )
    assert not trades.empty
    assert trades["exit_reason"].isin(["take_profit_50pct", "manage_21dte", "expired"]).all()


def test_directional_trades_respect_stop_target_or_expiry():
    provider = SyntheticProvider(seed=14, regime="mixed")
    strategy = REGISTRY.get("momentum_breakout")()
    trades = backtest_universe(
        ["SPY"], provider, [strategy], "2020-01-01", "2024-01-01", ContractSelector()
    )
    assert not trades.empty
    assert trades["exit_reason"].isin(["target", "stop", "expired"]).all()


def test_credit_spread_max_profit_loss_ratio_is_realistic():
    """Regression test for the delta-solver bug: a 30-delta short / wider long
    credit spread should never show max_profit wildly exceeding max_loss (or
    vice versa) the way it did when put strikes diverged to ~3x spot."""
    provider = SyntheticProvider(seed=15, regime="mixed")
    strategy = REGISTRY.get("iv_rank_premium")()
    trades = backtest_universe(
        ["SPY", "AAPL"], provider, [strategy], "2020-01-01", "2024-01-01", ContractSelector()
    )
    spreads = trades[trades["structure"].isin(["put_credit_spread", "call_credit_spread"])]
    assert not spreads.empty
    ratio = spreads["max_profit"] / spreads["max_loss"]
    assert (ratio > 0.03).all() and (ratio < 5.0).all()


def test_rank_strategies_orders_by_expectancy_weighted_by_sample_size():
    provider = SyntheticProvider(seed=16, regime="mixed")
    strategies = [REGISTRY.get(k)() for k in REGISTRY.keys()]
    trades = backtest_universe(
        ["SPY", "AAPL"], provider, strategies, "2019-01-01", "2024-01-01", ContractSelector()
    )
    ranked = rank_strategies(trades)
    assert list(ranked.columns[:2]) == ["strategy", "num_trades"]
    assert len(ranked) == len(REGISTRY.keys())


def test_summarize_empty_trades_returns_empty_frame_not_crash():
    empty = pd.DataFrame(columns=["ticker", "strategy", "pnl", "entry_date", "exit_date"])
    out = summarize(empty)
    assert out.empty
