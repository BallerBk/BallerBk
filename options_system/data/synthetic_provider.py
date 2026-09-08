"""Synthetic OHLCV generator.

Used for local validation of the engine in environments without market-data
network access (this sandbox included), and for stress-testing strategies
against regimes real history may not contain (pure chop, strong trend, high
vol). NOT a substitute for real data when you actually run the scanner live
-- swap in YFinanceProvider or your broker's data feed for that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from options_system.data.base import PriceDataProvider, validate_ohlcv


class SyntheticProvider(PriceDataProvider):
    def __init__(
        self,
        seed: int | None = 42,
        annual_drift: float = 0.08,
        annual_vol: float = 0.28,
        start_price: float = 100.0,
        regime: str = "mixed",
    ):
        """
        regime: 'trend_up', 'trend_down', 'chop', 'mixed'
            Controls the drift/mean-reversion mix so backtests can be sanity
            checked against a regime a strategy is *expected* to win or lose in.
        """
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.start_price = start_price
        self.regime = regime

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed_for(ticker))
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n == 0:
            return validate_ohlcv(
                pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), ticker
            )

        dt = 1 / 252
        sigma = self.annual_vol
        mu = self._drift_for_regime(n)

        shocks = rng.normal(0, 1, n)
        if self.regime == "chop":
            # mean-reverting around start_price via an OU-ish nudge
            log_prices = np.zeros(n)
            log_prices[0] = np.log(self.start_price)
            theta = 3.0
            mean_log = np.log(self.start_price)
            for i in range(1, n):
                log_prices[i] = (
                    log_prices[i - 1]
                    + theta * (mean_log - log_prices[i - 1]) * dt
                    + sigma * np.sqrt(dt) * shocks[i]
                )
        else:
            log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
            log_prices = np.log(self.start_price) + np.cumsum(log_returns)

        close = np.exp(log_prices)
        # Build plausible OHLC around each close using intraday noise
        intraday_range = close * sigma * np.sqrt(dt) * rng.uniform(0.4, 1.2, n)
        open_ = np.roll(close, 1)
        open_[0] = self.start_price
        high = np.maximum(open_, close) + intraday_range * rng.uniform(0.1, 0.6, n)
        low = np.minimum(open_, close) - intraday_range * rng.uniform(0.1, 0.6, n)
        volume = rng.integers(1_000_000, 8_000_000, n)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        return validate_ohlcv(df, ticker)

    def _seed_for(self, ticker: str) -> int:
        if self.seed is None:
            return np.random.randint(0, 2**31 - 1)
        return (self.seed + sum(ord(c) for c in ticker)) % (2**31 - 1)

    def _drift_for_regime(self, n: int) -> np.ndarray:
        if self.regime == "trend_up":
            return np.full(n, abs(self.annual_drift) + 0.10)
        if self.regime == "trend_down":
            return np.full(n, -abs(self.annual_drift) - 0.10)
        if self.regime == "chop":
            return np.zeros(n)  # unused directly, chop branch handles its own path
        # mixed: piecewise regime segments so a full backtest sees trend + chop
        segments = max(3, n // 60)
        bounds = np.linspace(0, n, segments + 1).astype(int)
        drift = np.zeros(n)
        rng = np.random.default_rng(self._seed_for("regime-mix"))
        for i in range(segments):
            choice = rng.choice(["up", "down", "flat"], p=[0.4, 0.25, 0.35])
            val = {"up": 0.35, "down": -0.30, "flat": 0.0}[choice]
            drift[bounds[i] : bounds[i + 1]] = val
        return drift
