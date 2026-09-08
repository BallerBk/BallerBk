"""Turn a strategy signal into a concrete, tradeable option contract idea.

This is the piece that answers "what should the option look like": call vs
put vs spread, how far out (DTE), and which strike(s) -- using Black-Scholes
delta-solving off an underlying price + volatility estimate, since a live
option chain isn't available in this environment. Every number here is an
estimate for planning purposes, not an executable quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from options_system.options.black_scholes import (
    price_and_greeks,
    round_to_strike_increment,
    strike_for_target_delta,
)
from options_system.strategies.base import BEARISH, BULLISH, NEUTRAL_RANGE

RISK_FREE_RATE_DEFAULT = 0.045  # approx short-term T-bill yield; override via ContractSelector(r=...)


@dataclass
class ContractIdea:
    structure: str  # long_call | long_put | put_credit_spread | call_credit_spread | iron_condor
    dte: int
    expiration_note: str
    legs: list[dict] = field(default_factory=list)  # [{"side": "buy"/"sell", "type": "call"/"put", "strike": ..., "delta": ..., "est_price": ...}]
    est_net_debit_credit: float = 0.0  # positive = net debit paid, negative = net credit received
    max_loss: float | None = None
    max_profit: float | None = None
    notes: str = ""


def default_strike_increment(price: float) -> float:
    if price < 25:
        return 0.5
    if price < 100:
        return 1.0
    if price < 300:
        return 2.5
    return 5.0


def structure_for(strategy_default: str, direction: str) -> str:
    if strategy_default == "long_call":
        if direction == BULLISH:
            return "long_call"
        if direction == BEARISH:
            return "long_put"
        return "iron_condor"  # shouldn't happen for directional strategies
    if strategy_default in ("credit_spread", "iron_condor"):
        if direction == BULLISH:
            return "put_credit_spread"
        if direction == BEARISH:
            return "call_credit_spread"
        if direction == NEUTRAL_RANGE:
            return "iron_condor"
    raise ValueError(f"Unhandled combination: {strategy_default=} {direction=}")


class ContractSelector:
    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE_DEFAULT,
        directional_dte: int = 30,
        premium_selling_dte: int = 45,
        directional_target_delta: float = 0.60,
        credit_spread_short_delta: float = 0.30,
        iron_condor_wing_delta: float = 0.16,
        width_std_mult: float = 0.5,
    ):
        self.r = risk_free_rate
        self.directional_dte = directional_dte
        self.premium_selling_dte = premium_selling_dte
        self.directional_target_delta = directional_target_delta
        self.credit_spread_short_delta = credit_spread_short_delta
        self.iron_condor_wing_delta = iron_condor_wing_delta
        # Spread width is a fraction of the expected move (s * sigma * sqrt(t)),
        # not a fixed % of price -- a fixed % badly distorts risk/reward across
        # volatility regimes (a wide fixed-$ spread on a low-vol name prices the
        # long leg near zero, making max_profit ~= width instead of a sane credit).
        self.width_std_mult = width_std_mult

    def build(
        self,
        underlying_price: float,
        sigma: float,
        strategy_default_structure: str,
        direction: str,
        dte_override: int | None = None,
    ) -> ContractIdea:
        structure = structure_for(strategy_default_structure, direction)
        increment = default_strike_increment(underlying_price)
        sigma = max(sigma, 0.05)  # floor to avoid degenerate BS solves on near-zero vol

        if structure in ("long_call", "long_put"):
            dte = dte_override or self.directional_dte
            return self._build_directional(underlying_price, sigma, dte, structure, increment)
        if structure in ("put_credit_spread", "call_credit_spread"):
            dte = dte_override or self.premium_selling_dte
            return self._build_credit_spread(underlying_price, sigma, dte, structure, increment)
        if structure == "iron_condor":
            dte = dte_override or self.premium_selling_dte
            return self._build_iron_condor(underlying_price, sigma, dte, increment)
        raise ValueError(f"Unknown structure {structure}")

    def _build_directional(self, s, sigma, dte, structure, increment) -> ContractIdea:
        opt_type = "call" if structure == "long_call" else "put"
        t = dte / 365
        raw_strike = strike_for_target_delta(
            s, t, self.r, sigma, opt_type, self.directional_target_delta
        )
        strike = round_to_strike_increment(raw_strike, increment)
        g = price_and_greeks(s, strike, t, self.r, sigma, opt_type)
        return ContractIdea(
            structure=structure,
            dte=dte,
            expiration_note=f"~{dte} DTE (target ~{self.directional_target_delta:.0%} delta)",
            legs=[{"side": "buy", "type": opt_type, "strike": strike, "delta": round(g.delta, 3), "est_price": round(g.price, 2)}],
            est_net_debit_credit=round(g.price, 2),
            max_loss=round(g.price * 100, 2),
            max_profit=None,  # theoretically unlimited (call) / large (put)
            notes="Directional debit trade. Max loss = premium paid x100/contract. Consider a debit spread to cap cost if desired.",
        )

    def _build_credit_spread(self, s, sigma, dte, structure, increment) -> ContractIdea:
        opt_type = "put" if structure == "put_credit_spread" else "call"
        t = dte / 365
        short_raw = strike_for_target_delta(
            s, t, self.r, sigma, opt_type, self.credit_spread_short_delta
        )
        short_strike = round_to_strike_increment(short_raw, increment)
        expected_move = s * sigma * (t**0.5)
        width = round_to_strike_increment(expected_move * self.width_std_mult, increment)
        width = max(width, increment)
        long_strike = short_strike - width if opt_type == "put" else short_strike + width

        g_short = price_and_greeks(s, short_strike, t, self.r, sigma, opt_type)
        g_long = price_and_greeks(s, long_strike, t, self.r, sigma, opt_type)
        net_credit = g_short.price - g_long.price
        max_loss = max(width - net_credit, 0) * 100
        max_profit = net_credit * 100

        return ContractIdea(
            structure=structure,
            dte=dte,
            expiration_note=f"~{dte} DTE (tastytrade-style, short strike ~{self.credit_spread_short_delta:.0%} delta)",
            legs=[
                {"side": "sell", "type": opt_type, "strike": short_strike, "delta": round(g_short.delta, 3), "est_price": round(g_short.price, 2)},
                {"side": "buy", "type": opt_type, "strike": long_strike, "delta": round(g_long.delta, 3), "est_price": round(g_long.price, 2)},
            ],
            est_net_debit_credit=round(-net_credit, 2),
            max_loss=round(max_loss, 2),
            max_profit=round(max_profit, 2),
            notes="Defined-risk credit spread. Plan: take profit at 50% of max credit; consider closing/rolling at 21 DTE per tastytrade's published guidance.",
        )

    def _build_iron_condor(self, s, sigma, dte, increment) -> ContractIdea:
        t = dte / 365
        put_short_raw = strike_for_target_delta(s, t, self.r, sigma, "put", self.iron_condor_wing_delta)
        call_short_raw = strike_for_target_delta(s, t, self.r, sigma, "call", self.iron_condor_wing_delta)
        put_short = round_to_strike_increment(put_short_raw, increment)
        call_short = round_to_strike_increment(call_short_raw, increment)
        expected_move = s * sigma * (t**0.5)
        width = max(round_to_strike_increment(expected_move * self.width_std_mult, increment), increment)
        put_long = put_short - width
        call_long = call_short + width

        g_ps = price_and_greeks(s, put_short, t, self.r, sigma, "put")
        g_pl = price_and_greeks(s, put_long, t, self.r, sigma, "put")
        g_cs = price_and_greeks(s, call_short, t, self.r, sigma, "call")
        g_cl = price_and_greeks(s, call_long, t, self.r, sigma, "call")

        net_credit = (g_ps.price - g_pl.price) + (g_cs.price - g_cl.price)
        max_loss = max(width - net_credit, 0) * 100
        max_profit = net_credit * 100

        return ContractIdea(
            structure="iron_condor",
            dte=dte,
            expiration_note=f"~{dte} DTE, short strikes ~{self.iron_condor_wing_delta:.0%} delta each side",
            legs=[
                {"side": "sell", "type": "put", "strike": put_short, "delta": round(g_ps.delta, 3), "est_price": round(g_ps.price, 2)},
                {"side": "buy", "type": "put", "strike": put_long, "delta": round(g_pl.delta, 3), "est_price": round(g_pl.price, 2)},
                {"side": "sell", "type": "call", "strike": call_short, "delta": round(g_cs.delta, 3), "est_price": round(g_cs.price, 2)},
                {"side": "buy", "type": "call", "strike": call_long, "delta": round(g_cl.delta, 3), "est_price": round(g_cl.price, 2)},
            ],
            est_net_debit_credit=round(-net_credit, 2),
            max_loss=round(max_loss, 2),
            max_profit=round(max_profit, 2),
            notes="Range-bound premium sale. Plan: take profit at 50% of max credit; manage/close at 21 DTE per tastytrade's published guidance.",
        )
