from options_system.data.synthetic_provider import SyntheticProvider
from options_system.options.contract_selector import ContractSelector
from options_system.scanner import candidates_to_dataframe, scan_market
from options_system.strategies import REGISTRY


def test_scan_market_returns_candidates_sorted_by_confidence():
    provider = SyntheticProvider(seed=21, regime="mixed")
    candidates = scan_market(
        ["SPY", "AAPL", "MSFT", "NVDA"], provider, list(REGISTRY.keys()), ContractSelector()
    )
    confidences = [c.confidence for c in candidates]
    assert confidences == sorted(confidences, reverse=True)


def test_scan_market_skips_unknown_ticker_gracefully():
    class BrokenProvider(SyntheticProvider):
        def get_history_recent(self, ticker, lookback_days):
            if ticker == "BROKEN":
                raise ValueError("no data")
            return super().get_history_recent(ticker, lookback_days)

    provider = BrokenProvider(seed=22)
    candidates = scan_market(["SPY", "BROKEN"], provider, list(REGISTRY.keys()), ContractSelector())
    # should not raise, and SPY should still be able to produce candidates
    assert all(c.ticker != "BROKEN" for c in candidates)


def test_candidates_to_dataframe_has_expected_columns():
    provider = SyntheticProvider(seed=23, regime="mixed")
    candidates = scan_market(["SPY", "QQQ"], provider, list(REGISTRY.keys()), ContractSelector())
    df = candidates_to_dataframe(candidates)
    if not df.empty:
        assert "ticker" in df.columns
        assert "structure" in df.columns
        assert "confidence" in df.columns
