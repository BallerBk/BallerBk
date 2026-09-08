from options_system.strategies.base import Strategy, StrategyRegistry
from options_system.strategies.iv_rank_premium import IvRankPremiumStrategy
from options_system.strategies.momentum_breakout import MomentumBreakoutStrategy
from options_system.strategies.range_reversion import RangeReversionStrategy
from options_system.strategies.trend_pullback import TrendPullbackStrategy

REGISTRY = StrategyRegistry()
REGISTRY.register(TrendPullbackStrategy)
REGISTRY.register(MomentumBreakoutStrategy)
REGISTRY.register(RangeReversionStrategy)
REGISTRY.register(IvRankPremiumStrategy)

__all__ = [
    "Strategy",
    "StrategyRegistry",
    "REGISTRY",
    "TrendPullbackStrategy",
    "MomentumBreakoutStrategy",
    "RangeReversionStrategy",
    "IvRankPremiumStrategy",
]
