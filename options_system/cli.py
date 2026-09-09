"""Command-line entry point.

Usage:
    python -m options_system.cli backtest [--config config.yaml] [--strategy KEY]
    python -m options_system.cli scan     [--config config.yaml] [--out candidates.json]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from options_system.backtest.engine import backtest_universe
from options_system.backtest.metrics import rank_strategies, summarize
from options_system.backtest.orb_engine import backtest_orb_universe
from options_system.config import load_config
from options_system.data.csv_provider import CsvProvider
from options_system.data.synthetic_provider import SyntheticProvider
from options_system.options.contract_selector import ContractSelector
from options_system.scanner import candidates_to_dataframe, save_candidates_json, scan_market
from options_system.strategies import REGISTRY


def _split_daily_and_intraday(strategy_keys: list[str]) -> tuple[list[str], list[str]]:
    daily, intraday = [], []
    for k in strategy_keys:
        (intraday if REGISTRY.get(k).intraday else daily).append(k)
    return daily, intraday


def _build_provider(cfg: dict):
    kind = cfg.get("data_provider", "synthetic")
    if kind == "synthetic":
        print("[options-system] Using SYNTHETIC data (no real market data access in "
              "this environment). Switch data_provider to 'yfinance' or 'csv' in "
              "config.yaml when running with real internet access.")
        return SyntheticProvider(regime="mixed")
    if kind == "yfinance":
        from options_system.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    if kind == "csv":
        return CsvProvider(cfg["csv_data_dir"])
    raise ValueError(f"Unknown data_provider '{kind}' in config.yaml")


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    provider = _build_provider(cfg)
    selector = ContractSelector(risk_free_rate=cfg["risk_free_rate"])

    strategy_keys = [args.strategy] if args.strategy else cfg["strategies"]
    daily_keys, intraday_keys = _split_daily_and_intraday(strategy_keys)
    if intraday_keys:
        print(f"[options-system] Skipping intraday strategy/strategies {intraday_keys} here -- "
              f"they need minute bars and same-day exits. Use: python -m options_system.cli orb-backtest")
    if not daily_keys:
        print("No daily-bar strategies selected. Nothing to backtest.")
        return
    strategies = [REGISTRY.get(k)() for k in daily_keys]

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=cfg["backtest_lookback_years"])

    trades = backtest_universe(
        cfg["universe"], provider, strategies, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), selector
    )

    if trades.empty:
        print("No trades were generated. Try a longer lookback or different strategies.")
        return

    if args.out:
        trades.to_csv(args.out, index=False)
        print(f"Wrote {len(trades)} trades to {args.out}")

    print("\n=== Strategy ranking (by trade expectancy, tie-broken by sample size) ===")
    print(rank_strategies(trades).to_string(index=False))

    print("\n=== Per-ticker breakdown ===")
    print(summarize(trades, group_by="ticker").to_string(index=False))


def cmd_scan(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    provider = _build_provider(cfg)
    selector = ContractSelector(risk_free_rate=cfg["risk_free_rate"])

    daily_keys, intraday_keys = _split_daily_and_intraday(cfg["strategies"])
    if intraday_keys:
        print(f"[options-system] Skipping intraday strategy/strategies {intraday_keys} in the "
              f"once-daily scan -- they need live minute bars during market hours, not a "
              f"once-daily post-close scan. Use: python -m options_system.cli orb-backtest")
    candidates = scan_market(cfg["universe"], provider, daily_keys, selector)

    if not candidates:
        print("No trade candidates today across the configured universe/strategies.")
        return

    df = candidates_to_dataframe(candidates)
    print(f"\n=== {len(candidates)} trade candidate(s) ===")
    print(df.to_string(index=False))

    out_path = args.out or "candidates.json"
    save_candidates_json(candidates, out_path)
    print(f"\nSaved full detail to {out_path}")
    print(
        "\nReminder: these are model-generated ideas for review, not trade "
        "instructions -- verify strikes/prices against a live option chain "
        "before acting on anything."
    )


def cmd_orb_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    provider = _build_provider(cfg)

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=args.days)

    trades = backtest_orb_universe(
        cfg["universe"], provider, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        bar_minutes=args.bar_minutes,
    )

    if trades.empty:
        print("No ORB trades were generated. Try more days, or check the data provider "
              "supports get_intraday_history (yfinance's intraday history is limited to "
              "roughly the trailing 60 days regardless of --days).")
        return

    if args.out:
        trades.to_csv(args.out, index=False)
        print(f"Wrote {len(trades)} trades to {args.out}")

    trades["minutes_held"] = (
        pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
    ).dt.total_seconds() / 60

    print("\n=== ORB (0DTE) results ===")
    print(summarize(trades, group_by="ticker").to_string(index=False))
    print(f"\nAvg minutes held: {trades['minutes_held'].mean():.0f}  "
          f"(same-day exits only -- 'avg_days_held' above is not meaningful for this strategy)")
    print(
        "\nReminder: 0DTE options decay and move fast. This backtest uses Black-Scholes "
        "with volatility held constant per trade from an intraday realized-vol estimate -- "
        "see README's ORB section for what that does and doesn't capture."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="options-system")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="Backtest strategies over the configured universe")
    p_bt.add_argument("--strategy", choices=REGISTRY.keys(), help="Limit to a single strategy")
    p_bt.add_argument("--out", help="CSV path to save the raw trade log")
    p_bt.set_defaults(func=cmd_backtest)

    p_scan = sub.add_parser("scan", help="Scan the universe for today's trade candidates")
    p_scan.add_argument("--out", help="JSON path to save candidates (default candidates.json)")
    p_scan.set_defaults(func=cmd_scan)

    p_orb = sub.add_parser(
        "orb-backtest",
        help="Backtest Opening Range Breakout (0DTE, intraday) separately from the daily strategies",
    )
    p_orb.add_argument("--days", type=int, default=30, help="Lookback window in calendar days")
    p_orb.add_argument("--bar-minutes", type=int, default=5, help="Intraday bar size in minutes")
    p_orb.add_argument("--out", help="CSV path to save the raw trade log")
    p_orb.set_defaults(func=cmd_orb_backtest)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
