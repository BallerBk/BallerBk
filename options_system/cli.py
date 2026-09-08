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
from options_system.config import load_config
from options_system.data.csv_provider import CsvProvider
from options_system.data.synthetic_provider import SyntheticProvider
from options_system.options.contract_selector import ContractSelector
from options_system.scanner import candidates_to_dataframe, save_candidates_json, scan_market
from options_system.strategies import REGISTRY


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
    strategies = [REGISTRY.get(k)() for k in strategy_keys]

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

    candidates = scan_market(cfg["universe"], provider, cfg["strategies"], selector)

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
