import pandas as pd
import pytest

from options_system.data.synthetic_provider import SyntheticProvider
from options_system.strategies import REGISTRY
from options_system.strategies.base import NONE, SIGNAL_COLUMNS


@pytest.fixture(scope="module")
def price_history():
    provider = SyntheticProvider(seed=1, regime="mixed")
    return provider.get_history("TEST", "2020-01-01", "2024-01-01")


@pytest.mark.parametrize("key", REGISTRY.keys())
def test_strategy_returns_aligned_signal_frame(key, price_history):
    strategy = REGISTRY.get(key)()
    signals = strategy.generate_signals(price_history)
    assert list(signals.index) == list(price_history.index)
    assert set(SIGNAL_COLUMNS).issubset(signals.columns)


@pytest.mark.parametrize("key", REGISTRY.keys())
def test_strategy_produces_at_least_one_signal_over_4_years(key, price_history):
    strategy = REGISTRY.get(key)()
    signals = strategy.generate_signals(price_history)
    fired = signals[signals["direction"] != NONE]
    assert len(fired) > 0, f"{key} never fired a signal over 4 years of mixed-regime data"


@pytest.mark.parametrize("key", REGISTRY.keys())
def test_confidence_is_bounded(key, price_history):
    strategy = REGISTRY.get(key)()
    signals = strategy.generate_signals(price_history)
    conf = signals["confidence"].astype(float)
    assert (conf >= 0).all() and (conf <= 1).all()


def test_short_history_returns_no_signals_without_crashing():
    provider = SyntheticProvider(seed=2)
    short_df = provider.get_history("TEST", "2024-01-01", "2024-01-20")
    for key in REGISTRY.keys():
        strategy = REGISTRY.get(key)()
        signals = strategy.generate_signals(short_df)
        assert (signals["direction"] == NONE).all()


def test_trend_pullback_bullish_has_stop_below_and_target_above_entry():
    from options_system.strategies.trend_pullback import TrendPullbackStrategy

    provider = SyntheticProvider(seed=3, regime="trend_up")
    df = provider.get_history("TEST", "2020-01-01", "2022-01-01")
    strategy = TrendPullbackStrategy()
    signals = strategy.generate_signals(df)
    bullish = signals[signals["direction"] == "bullish"]
    assert len(bullish) > 0
    merged = bullish.join(df["close"])
    assert (merged["stop_price"] < merged["close"]).all()
    assert (merged["target_price"] > merged["close"]).all()
