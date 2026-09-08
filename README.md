# Options System

A research/decision-support system for options trading: a Python backtesting
+ market-scanning engine with four selectable strategies, plus a companion
TradingView Pine Script indicator for charting/alerts.

**This is not connected to a broker. It does not place trades. It outputs
ideas for you to review, never instructions to execute automatically.**

## Why this exists / how it was built

This was built from **publicly documented, generic trading concepts** —
tastytrade's published IV-rank premium-selling research, and standard
mechanical breakout/trend/mean-reversion rules that appear throughout public
technical-analysis literature. It is **not** a reproduction of any specific
paid course or educator's proprietary system — that material (Discord
memberships, gated course dashboards, paid coaching) was explicitly kept out
of scope. See the strategy modules' docstrings in `options_system/strategies/`
for the specific public source of each rule set.

## Architecture

```
options_system/
  data/            Pluggable price-data providers (synthetic / yfinance / CSV)
  indicators.py    Shared technical indicators (EMA, RSI, ATR, Bollinger, Donchian, vol rank)
  strategies/      Four selectable strategies, common Strategy interface
  options/         Black-Scholes pricing + delta-targeted contract selection
  backtest/        Event-driven backtester + performance metrics/ranking
  scanner.py       Scans a ticker universe, emits trade candidates
  cli.py           `backtest` and `scan` commands
pine/
  multi_strategy_signals.pine   TradingView indicator mirroring the 4 strategies
tests/             pytest suite (45 tests) validating pricing, strategies, backtester
config.yaml        Ticker universe, active strategies, risk settings
```

### The four strategies (selectable independently or all at once)

| Strategy | Key | Concept | Structure |
|---|---|---|---|
| Trend Pullback Continuation | `trend_pullback` | Buy pullbacks to the fast EMA within an EMA trend, RSI-filtered | Long call/put (debit) |
| Momentum Breakout | `momentum_breakout` | N-day high/low breakout confirmed by volume ≥1.2x average | Long call/put (debit) |
| Range / Support-Resistance Reversion | `range_reversion` | Fade Bollinger Band touches in a flat-trend (range-bound) market | Credit spread |
| IV-Rank Premium Selling | `iv_rank_premium` | Sell premium when realized-vol rank (IV Rank proxy) > 50, tastytrade-style: ~45 DTE, ~30Δ short strike, 50% take-profit, manage at 21 DTE | Credit spread / iron condor |

Each strategy is independently backtestable and ranked by expectancy, so
`options-system backtest` tells you which one(s) are actually working on
your chosen universe/timeframe rather than assuming any single approach.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml`: set your ticker universe, which strategies are active,
account size / risk %, and **data_provider**.

### Data source — read this first

**This sandbox environment has no network access to any market-data
provider** (Yahoo Finance, Alpha Vantage, Polygon, etc. are all blocked at
the network level here). Everything in this repo was built and tested
against `SyntheticProvider` — a random-walk price generator used purely to
validate that the engine's logic is correct (signal generation, option
pricing, exit rules, P&L accounting). **Synthetic data proves the machinery
works; it does not prove the strategies have a real edge** — see
"Backtest limitations" below.

To run this for real, on a machine with normal internet access:

```yaml
data_provider: yfinance   # in config.yaml
```

```bash
pip install yfinance
python -m options_system.cli backtest
python -m options_system.cli scan
```

Or point `data_provider: csv` at your own exported OHLCV history in
`csv_data_dir` if you'd rather not depend on Yahoo Finance.

## Usage

```bash
# Backtest all configured strategies over the configured universe
python -m options_system.cli backtest

# Backtest just one strategy
python -m options_system.cli backtest --strategy iv_rank_premium

# Scan for today's trade candidates (prints a table + writes candidates.json)
python -m options_system.cli scan
```

`scan` output per candidate: ticker, strategy, direction, option structure
(long_call / long_put / put_credit_spread / call_credit_spread /
iron_condor), legs (side/type/strike), estimated net debit or credit, max
profit/loss, DTE, and a confidence score. **Verify every strike/price
against a live option chain before acting — these are Black-Scholes
estimates off historical volatility, not real quotes.**

### How often to run the scanner

All four strategies operate on **daily bars** with 30–45 DTE holding
horizons (matching the published cadence of the research they're built on).
Run `scan` **once per day**, after market close (using that day's completed
bar) — set `scan_cadence` in `config.yaml`. Running it intraday adds noise,
not edge, for this rule set. If you want an intraday/0DTE opening-range
strategy later, that would be a fifth, separate strategy module with its
own cadence — ask and I'll add it.

## TradingView companion (`pine/multi_strategy_signals.pine`)

Paste into TradingView's Pine Editor and add to a chart. It mirrors the same
four strategies' *technical* logic (it can't see the options chain — nothing
on TradingView's free script API can) with a dropdown to pick which
strategy's signals to display, plus `alertcondition()`s for every signal so
you can wire up TradingView alerts. Use it for live chart visualization;
use the Python `scan` command for the actual call/put/strike/DTE
recommendation.

**I have not been able to compile-test this script against TradingView
directly** (no browser/TradingView access from this environment) — I
verified the Pine v5 syntax by hand. If the Pine Editor throws a compile
error when you paste it in, send me the exact error message and I'll fix it
immediately.

## Backtest limitations (read before trusting any number)

1. **Volatility path**: each simulated trade holds volatility constant at
   its entry-day realized-vol estimate. Real IV moves throughout a trade's
   life (mean-reverts, spikes around earnings, etc.) — this backtester does
   not model that.
2. **IV Rank proxy**: `iv_rank_premium` uses realized (historical)
   volatility rank as a stand-in for true IV Rank, because free historical
   *implied* volatility isn't available without a paid data feed. Real IV
   is usually richer than realized vol (the volatility risk premium) —
   which is the actual structural edge behind premium selling in live
   markets. A realized-vol proxy does not capture that premium, so backtest
   results for this strategy are conservative at best and shouldn't be
   read as "the strategy doesn't work" if they look weak on synthetic data.
3. **Synthetic data has no volatility risk premium at all** — it's a random
   walk with regime-switching drift, built to exercise the code paths, not
   to model real market microstructure. Numbers from `SyntheticProvider`
   validate the *engine*; they say nothing about real-world edge. Re-run
   `backtest` with `data_provider: yfinance` (or your own CSV history)
   before drawing any conclusion about which strategy to actually trade.
4. **No slippage/commissions**: fills are assumed at the Black-Scholes mid.
   Real fills, especially on multi-leg spreads, will be worse.
5. **No dividends beyond a flat `q=0`** in the Black-Scholes model by
   default, and no early-assignment risk on American-style equity options.
6. **One position per (ticker, strategy) at a time** — the engine won't
   pyramid or overlap trades on the same ticker/strategy pair, by design
   (keeps the backtest interpretable), but this understates how a real
   portfolio might size across concurrent setups.

### Improving the IV-rank proxy

If you get access to a real implied-volatility data source (a paid feed, a
broker API that exposes the option chain, ORATS, etc.), swap
`indicators.iv_rank_proxy` for a function that reads real IV Rank, and the
rest of `iv_rank_premium.py` needs no other changes — it was built against
that interface deliberately.

## Extending

- **Add a strategy**: subclass `options_system.strategies.base.Strategy`,
  implement `generate_signals(df) -> DataFrame`, register it in
  `options_system/strategies/__init__.py`. The backtester, scanner, and CLI
  all pick it up automatically via the registry — no other code changes.
- **Add a data source**: subclass `options_system.data.base.PriceDataProvider`.
- **Position sizing**: `config.yaml`'s `risk_pct_per_trade` is currently
  informational only (shown for your own manual sizing) — the backtester
  reports P&L per single contract/spread, not portfolio-level sizing. Ask
  if you want Kelly-fraction or fixed-risk position sizing wired in.
- **Broker auto-execution**: out of scope by design (see top of this file).
  If you want it later, that's a separate, explicit build — alerts would
  need to go through a webhook bridge (e.g. TradingView alert → a hosted
  relay → your broker's API), and it carries real execution risk that a
  signal-only system doesn't.

## Disclaimers

Not financial advice. Options trading involves substantial risk of loss.
Past backtested performance — especially on synthetic data — does not
predict future results. Every number this system produces (strikes,
premiums, greeks, P&L) is a model estimate, not a live quote; verify
everything against a real option chain before acting. You are solely
responsible for any trading decisions you make.
