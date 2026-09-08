import math

import pytest

from options_system.options.black_scholes import price_and_greeks, strike_for_target_delta


def test_call_delta_between_0_and_1():
    g = price_and_greeks(s=100, k=100, t=30 / 365, r=0.045, sigma=0.25, option_type="call")
    assert 0 < g.delta < 1
    assert g.price > 0


def test_put_delta_between_minus1_and_0():
    g = price_and_greeks(s=100, k=100, t=30 / 365, r=0.045, sigma=0.25, option_type="put")
    assert -1 < g.delta < 0
    assert g.price > 0


def test_put_call_parity_roughly_holds():
    s, k, t, r, sigma = 100, 100, 30 / 365, 0.045, 0.25
    call = price_and_greeks(s, k, t, r, sigma, "call")
    put = price_and_greeks(s, k, t, r, sigma, "put")
    lhs = call.price - put.price
    rhs = s - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=0.05)


@pytest.mark.parametrize("target_delta", [0.10, 0.20, 0.30, 0.50, 0.70])
def test_call_strike_solver_converges_to_target_delta(target_delta):
    s, t, r, sigma = 150, 45 / 365, 0.045, 0.30
    strike = strike_for_target_delta(s, t, r, sigma, "call", target_delta)
    g = price_and_greeks(s, strike, t, r, sigma, "call")
    assert g.delta == pytest.approx(target_delta, abs=0.01)


@pytest.mark.parametrize("target_delta", [-0.10, -0.20, -0.30, -0.50, -0.70])
def test_put_strike_solver_converges_to_target_delta(target_delta):
    """Regression test: the solver previously had an inverted bisection
    direction for puts and diverged to the search boundary (~3x spot)
    instead of converging. This must stay pinned down."""
    s, t, r, sigma = 150, 45 / 365, 0.045, 0.30
    strike = strike_for_target_delta(s, t, r, sigma, "put", target_delta)
    g = price_and_greeks(s, strike, t, r, sigma, "put")
    assert g.delta == pytest.approx(target_delta, abs=0.01)
    # A put's delta magnitude should never require a strike near the search
    # boundary for a reasonable target -- catches boundary-divergence bugs.
    assert 0.3 * s < strike < 2.0 * s


def test_call_and_put_strike_ordering_makes_sense():
    """A higher target delta for a call should mean a LOWER strike (more ITM);
    same magnitude target delta for a put should mean deeper OTM (lower)."""
    s, t, r, sigma = 100, 45 / 365, 0.045, 0.25
    call_30 = strike_for_target_delta(s, t, r, sigma, "call", 0.30)
    call_60 = strike_for_target_delta(s, t, r, sigma, "call", 0.60)
    assert call_60 < call_30  # higher delta call = lower/more ITM strike

    put_10 = strike_for_target_delta(s, t, r, sigma, "put", -0.10)
    put_30 = strike_for_target_delta(s, t, r, sigma, "put", -0.30)
    assert put_10 < put_30  # smaller |delta| put = further OTM = lower strike
    assert put_10 < s and put_30 < s  # both OTM puts sit below spot
