"""Common interface every strategy implements.

A strategy's only job is: look at a price history, decide per-bar whether
there's a directional/volatility edge, and describe it. It does NOT price
options or size positions -- that's options/contract_selector.py and
backtest/engine.py's job respectively. Keeping these separated is what lets
the backtester treat all four strategies identically and rank them
apples-to-apples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

# Direction values
BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL_RANGE = "neutral_range"  # range-bound: sell premium both sides (iron condor)
NONE = "none"

SIGNAL_COLUMNS = ["direction", "confidence", "reason", "stop_price", "target_price"]


@dataclass
class StrategyParams:
    """Base class for per-strategy tunable parameters. Subclass and extend."""

    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    #: unique key used in config.yaml and the CLI --strategy flag
    key: str = "base"
    #: human-readable name shown in scanner output and Pine Script dropdown
    display_name: str = "Base Strategy"
    #: one-line description of the public, well-documented concept it implements
    description: str = ""
    #: default option structure this strategy trades when direction is bullish/bearish
    #: one of: "long_call"/"long_put" (debit, directional), "credit_spread" (defined-risk
    #: premium selling), "iron_condor" (range-bound premium selling)
    default_structure: str = "long_call"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame aligned to df.index with columns SIGNAL_COLUMNS.

        direction: one of BULLISH, BEARISH, NEUTRAL_RANGE, NONE
        confidence: float in [0, 1], relative conviction for this bar's setup
        reason: short human-readable string of which rule fired
        stop_price / target_price: underlying price levels for risk framing
            (used to size the option trade's DTE/holding horizon in the
            backtester, not a promise the option itself stops at that price)
        """
        raise NotImplementedError

    def empty_signals(self, index: pd.Index) -> pd.DataFrame:
        out = pd.DataFrame(index=index, columns=SIGNAL_COLUMNS)
        out["direction"] = NONE
        out["confidence"] = 0.0
        out["reason"] = ""
        out["stop_price"] = float("nan")
        out["target_price"] = float("nan")
        return out


class StrategyRegistry:
    def __init__(self):
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, cls: type[Strategy]) -> None:
        self._strategies[cls.key] = cls

    def get(self, key: str) -> type[Strategy]:
        if key not in self._strategies:
            raise KeyError(
                f"Unknown strategy '{key}'. Available: {sorted(self._strategies)}"
            )
        return self._strategies[key]

    def keys(self) -> list[str]:
        return sorted(self._strategies)

    def all(self) -> dict[str, type[Strategy]]:
        return dict(self._strategies)
