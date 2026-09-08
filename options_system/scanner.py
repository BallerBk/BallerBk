"""Market scanner: runs every enabled strategy over the configured ticker
universe and, for each ticker where the *latest* bar produces a signal,
emits a concrete trade candidate (ticker, call/put/spread structure, DTE,
strikes, confidence).

This does not place any trade. It prints/saves candidates for you to review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from options_system.indicators import realized_vol
from options_system.options.contract_selector import ContractSelector
from options_system.strategies import REGISTRY
from options_system.strategies.base import NONE, Strategy


@dataclass
class Candidate:
    as_of: str
    ticker: str
    strategy: str
    strategy_name: str
    direction: str
    structure: str
    dte: int
    legs: list
    est_net_debit_credit: float
    max_profit: float | None
    max_loss: float | None
    confidence: float
    reason: str


def scan_market(
    tickers: list[str],
    data_provider,
    strategy_keys: list[str],
    selector: ContractSelector | None = None,
    lookback_days: int = 400,
    vol_length: int = 20,
) -> list[Candidate]:
    selector = selector or ContractSelector()
    strategies: list[Strategy] = [REGISTRY.get(k)() for k in strategy_keys]

    candidates: list[Candidate] = []
    for ticker in tickers:
        try:
            df = data_provider.get_history_recent(ticker, lookback_days)
        except Exception as e:
            print(f"[scanner] skipping {ticker}: could not load data ({e})")
            continue
        if df.empty or len(df) < 30:
            continue

        price = float(df["close"].iloc[-1])
        sigma_series = realized_vol(df["close"], length=vol_length)
        sigma = float(sigma_series.iloc[-1]) if pd.notna(sigma_series.iloc[-1]) else 0.20
        as_of = df.index[-1].strftime("%Y-%m-%d")

        for strategy in strategies:
            signals = strategy.generate_signals(df)
            last = signals.iloc[-1]
            if last["direction"] == NONE:
                continue

            idea = selector.build(price, sigma, strategy.default_structure, last["direction"])
            candidates.append(
                Candidate(
                    as_of=as_of,
                    ticker=ticker,
                    strategy=strategy.key,
                    strategy_name=strategy.display_name,
                    direction=last["direction"],
                    structure=idea.structure,
                    dte=idea.dte,
                    legs=idea.legs,
                    est_net_debit_credit=idea.est_net_debit_credit,
                    max_profit=idea.max_profit,
                    max_loss=idea.max_loss,
                    confidence=round(float(last["confidence"]), 3),
                    reason=str(last["reason"]),
                )
            )

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def candidates_to_dataframe(candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        leg_strs = [f"{l['side']} {l['type']} {l['strike']}" for l in c.legs]
        rows.append(
            {
                "as_of": c.as_of,
                "ticker": c.ticker,
                "strategy": c.strategy_name,
                "direction": c.direction,
                "structure": c.structure,
                "dte": c.dte,
                "legs": " / ".join(leg_strs),
                "net_debit(+)/credit(-)": c.est_net_debit_credit,
                "max_profit": c.max_profit,
                "max_loss": c.max_loss,
                "confidence": c.confidence,
                "reason": c.reason,
            }
        )
    return pd.DataFrame(rows)


def save_candidates_json(candidates: list[Candidate], path: str) -> None:
    with open(path, "w") as f:
        json.dump(
            {"generated_at": datetime.now().isoformat(), "candidates": [asdict(c) for c in candidates]},
            f,
            indent=2,
            default=str,
        )
