"""Black-Scholes pricing, greeks, and delta-targeted strike solving.

Standard textbook formulas (Black & Scholes 1973 / Merton). Used here for
two things:
  1. Estimating a strike that matches a target delta (e.g. "30-delta short
     strike", the tastytrade convention) when we only have an underlying
     price + volatility estimate, not a live option chain.
  2. Pricing simulated option positions in the backtester so P&L reflects
     time decay and IV level, not just underlying price movement.

This is a model, not a live quote. Real option prices differ from
Black-Scholes due to skew, term structure, dividends, and liquidity --
treat all outputs as estimates for planning/backtesting, not fill prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per 1 vol point (0.01)


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    if t <= 0 or sigma <= 0:
        raise ValueError("t and sigma must be positive")
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def price_and_greeks(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> Greeks:
    """
    s: underlying price, k: strike, t: years to expiration, r: risk-free rate,
    sigma: annualized volatility, option_type: 'call' or 'put', q: dividend yield.
    """
    if t <= 0:
        intrinsic = max(0.0, (s - k) if option_type == "call" else (k - s))
        return Greeks(price=intrinsic, delta=float(option_type == "call"), gamma=0.0, theta=0.0, vega=0.0)

    d1, d2 = _d1_d2(s, k, t, r - q, sigma)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)

    if option_type == "call":
        price = s * disc_q * norm.cdf(d1) - k * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        theta = (
            -(s * disc_q * norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
            - r * k * disc_r * norm.cdf(d2)
            + q * s * disc_q * norm.cdf(d1)
        ) / 365
    elif option_type == "put":
        price = k * disc_r * norm.cdf(-d2) - s * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -(s * disc_q * norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
            + r * k * disc_r * norm.cdf(-d2)
            - q * s * disc_q * norm.cdf(-d1)
        ) / 365
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = disc_q * norm.pdf(d1) / (s * sigma * math.sqrt(t))
    vega = s * disc_q * norm.pdf(d1) * math.sqrt(t) / 100  # per 1 vol point

    return Greeks(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega)


def strike_for_target_delta(
    s: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str,
    target_delta: float,
    q: float = 0.0,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Solve for the strike whose Black-Scholes delta matches target_delta.

    target_delta: for calls pass a positive value (e.g. 0.30), for puts pass
    a negative value (e.g. -0.30) OR a positive magnitude -- both are
    accepted and normalized to the right sign for the option type.
    """
    if option_type == "call" and target_delta < 0:
        target_delta = abs(target_delta)
    if option_type == "put" and target_delta > 0:
        target_delta = -target_delta

    lo, hi = s * 0.3, s * 3.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        d = price_and_greeks(s, mid, t, r, sigma, option_type, q).delta
        if abs(d - target_delta) < tol:
            return mid
        # Delta is monotonically decreasing in strike for BOTH calls (1 -> 0
        # as strike rises) and puts (0 -> -1 as strike rises, still a
        # decreasing raw value) -- so the same comparison direction is
        # correct for both. d > target means strike is too low (delta hasn't
        # fallen enough yet), so search the upper half.
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def round_to_strike_increment(price: float, increment: float = 1.0) -> float:
    return round(price / increment) * increment
