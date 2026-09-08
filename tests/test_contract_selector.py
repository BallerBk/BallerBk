import pytest

from options_system.options.contract_selector import ContractSelector, structure_for
from options_system.strategies.base import BEARISH, BULLISH, NEUTRAL_RANGE


def test_structure_for_directional():
    assert structure_for("long_call", BULLISH) == "long_call"
    assert structure_for("long_call", BEARISH) == "long_put"


def test_structure_for_premium_selling():
    assert structure_for("credit_spread", BULLISH) == "put_credit_spread"
    assert structure_for("credit_spread", BEARISH) == "call_credit_spread"
    assert structure_for("credit_spread", NEUTRAL_RANGE) == "iron_condor"


@pytest.mark.parametrize("direction", [BULLISH, BEARISH])
def test_credit_spread_risk_reward_is_sane(direction):
    """Regression test for the width-scaling bug: max_profit should never
    dwarf max_loss (or vice versa) for a standard 30-delta short / wider long
    credit spread -- that ratio should sit in a realistic band."""
    sel = ContractSelector()
    idea = sel.build(underlying_price=150, sigma=0.28, strategy_default_structure="credit_spread", direction=direction)
    assert idea.max_profit is not None and idea.max_loss is not None
    assert idea.max_profit > 0
    assert idea.max_loss > 0
    ratio = idea.max_profit / idea.max_loss
    assert 0.05 < ratio < 3.0, f"unrealistic risk/reward ratio {ratio}"


def test_iron_condor_has_four_legs_and_positive_credit():
    sel = ContractSelector()
    idea = sel.build(underlying_price=150, sigma=0.28, strategy_default_structure="iron_condor", direction=NEUTRAL_RANGE)
    assert len(idea.legs) == 4
    assert idea.est_net_debit_credit < 0  # negative = net credit received
    assert idea.max_profit > 0
    assert idea.max_loss > 0


def test_long_call_max_loss_equals_premium_paid():
    sel = ContractSelector()
    idea = sel.build(underlying_price=100, sigma=0.25, strategy_default_structure="long_call", direction=BULLISH)
    assert idea.structure == "long_call"
    assert idea.est_net_debit_credit > 0  # positive = net debit paid
    assert idea.max_loss == pytest.approx(idea.est_net_debit_credit * 100, rel=0.01)


def test_put_credit_spread_short_strike_below_long_strike_is_wrong_way_round():
    """For a put credit spread the short strike must be ABOVE the long strike
    (sell higher/closer-to-money put, buy lower/further-OTM put for protection)."""
    sel = ContractSelector()
    idea = sel.build(underlying_price=150, sigma=0.28, strategy_default_structure="credit_spread", direction=BULLISH)
    short_leg = next(l for l in idea.legs if l["side"] == "sell")
    long_leg = next(l for l in idea.legs if l["side"] == "buy")
    assert short_leg["strike"] > long_leg["strike"]
