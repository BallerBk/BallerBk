import pandas as pd
import pytest

from options_system.data.synthetic_provider import SyntheticProvider
from options_system.strategies.base import NONE
from options_system.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from options_system.strategies import REGISTRY
from options_system.backtest.orb_engine import backtest_orb_strategy, backtest_orb_universe


@pytest.fixture(scope="module")
def intraday_df():
    provider = SyntheticProvider(seed=31, regime="mixed")
    return provider.get_intraday_history("TEST", "2024-01-01", "2024-03-01", bar_minutes=5)


def test_orb_is_registered_and_flagged_intraday():
    assert "orb" in REGISTRY.keys()
    assert REGISTRY.get("orb").intraday is True
    # the other four must NOT be flagged intraday, or they'd get silently
    # skipped by the CLI's daily backtest/scan commands
    for key in REGISTRY.keys():
        if key != "orb":
            assert REGISTRY.get(key).intraday is False


def test_intraday_provider_returns_minute_bars_within_regular_session(intraday_df):
    assert not intraday_df.empty
    times = intraday_df.index.time
    assert all(t >= pd.Timestamp("09:30").time() for t in times)
    assert all(t < pd.Timestamp("16:00").time() for t in times)


def test_orb_signals_at_most_one_per_day(intraday_df):
    strategy = OpeningRangeBreakoutStrategy()
    signals = strategy.generate_signals(intraday_df)
    fired = signals[signals["direction"] != NONE]
    day_counts = fired.groupby(fired.index.normalize()).size()
    assert (day_counts <= 1).all()


def test_orb_signals_never_fire_during_the_opening_range_itself(intraday_df):
    strategy = OpeningRangeBreakoutStrategy(orb_minutes=15)
    signals = strategy.generate_signals(intraday_df)
    fired = signals[signals["direction"] != NONE]
    for ts in fired.index:
        day_open = ts.normalize() + pd.Timedelta(hours=9, minutes=30)
        assert ts >= day_open + pd.Timedelta(minutes=15)


def test_orb_stop_and_target_bracket_entry_correctly(intraday_df):
    strategy = OpeningRangeBreakoutStrategy()
    signals = strategy.generate_signals(intraday_df)
    merged = signals.join(intraday_df["close"], rsuffix="_price")
    fired = merged[merged["direction"] != NONE]
    bullish = fired[fired["direction"] == "bullish"]
    bearish = fired[fired["direction"] == "bearish"]
    assert (bullish["stop_price"] < bullish["close"]).all()
    assert (bullish["target_price"] > bullish["close"]).all()
    assert (bearish["stop_price"] > bearish["close"]).all()
    assert (bearish["target_price"] < bearish["close"]).all()


def test_orb_backtest_produces_same_day_trades_only(intraday_df):
    trades = backtest_orb_strategy("TEST", intraday_df)
    assert len(trades) > 0
    for t in trades:
        assert t.entry_date.normalize() == t.exit_date.normalize()
        assert t.exit_date >= t.entry_date


def test_orb_backtest_exit_reasons_are_valid(intraday_df):
    trades = backtest_orb_strategy("TEST", intraday_df)
    reasons = {t.exit_reason for t in trades}
    assert reasons.issubset({"target", "stop", "session_close"})


def test_orb_backtest_pnl_matches_option_premium_change(intraday_df):
    """pnl is computed once from full-precision entry/exit option prices;
    entry_option_price/exit_option_price stored on the trade are separately
    rounded to cents for display. Recomputing from those display-rounded
    fields can differ from the true pnl by up to ~$1/contract (two legs x
    up to half a cent x the 100 multiplier) -- that's expected double-
    rounding, not a bug, so the check is direction/magnitude sanity, not
    an exact recomputation."""
    trades = backtest_orb_strategy("TEST", intraday_df)
    assert len(trades) > 0
    for t in trades:
        approx_from_rounded_fields = (t.exit_option_price - t.entry_option_price) * 100
        assert t.pnl == pytest.approx(approx_from_rounded_fields, abs=1.0)
        # sign must agree (both positive, both negative, or both ~zero)
        assert (t.pnl > 0) == (approx_from_rounded_fields > 0) or abs(t.pnl) < 1.0


def test_orb_universe_skips_provider_without_intraday_support():
    class NoIntradayProvider(SyntheticProvider):
        pass  # inherits base's NotImplementedError for get_intraday_history... but SyntheticProvider overrides it

    # Use a provider that genuinely lacks intraday support (CsvProvider has no data on disk)
    from options_system.data.csv_provider import CsvProvider

    trades = backtest_orb_universe(["FAKE"], CsvProvider("/nonexistent"), start="2024-01-01", end="2024-02-01")
    assert trades.empty


def test_orb_universe_multiple_tickers(intraday_df):
    provider = SyntheticProvider(seed=32, regime="mixed")
    trades = backtest_orb_universe(["SPY", "AAPL"], provider, start="2024-01-01", end="2024-02-01")
    assert not trades.empty
    assert set(trades["ticker"].unique()).issubset({"SPY", "AAPL"})
    assert (trades["strategy"] == "orb").all()
