"""Performance metrics + strategy ranking from a trades DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, group_by: str = "strategy") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                group_by, "num_trades", "win_rate", "avg_pnl", "total_pnl",
                "expectancy", "profit_factor", "max_drawdown", "avg_days_held",
            ]
        )

    rows = []
    for key, g in trades.groupby(group_by):
        g = g.dropna(subset=["pnl"]).copy()
        if g.empty:
            continue
        wins = g[g["pnl"] > 0]["pnl"]
        losses = g[g["pnl"] <= 0]["pnl"]
        win_rate = len(wins) / len(g) if len(g) else 0.0
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        gross_win = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)

        equity = g.sort_values("exit_date")["pnl"].cumsum()
        running_max = equity.cummax()
        drawdown = (equity - running_max)
        max_dd = drawdown.min() if len(drawdown) else 0.0

        g["days_held"] = (pd.to_datetime(g["exit_date"]) - pd.to_datetime(g["entry_date"])).dt.days

        rows.append(
            {
                group_by: key,
                "num_trades": len(g),
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(g["pnl"].mean(), 2),
                "total_pnl": round(g["pnl"].sum(), 2),
                "expectancy": round(expectancy, 2),
                "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else profit_factor,
                "max_drawdown": round(max_dd, 2),
                "avg_days_held": round(g["days_held"].mean(), 1),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("expectancy", ascending=False).reset_index(drop=True)


def rank_strategies(trades: pd.DataFrame) -> pd.DataFrame:
    """Rank strategies by expectancy per trade, with num_trades as a confidence
    tiebreaker (a great expectancy over 3 trades is noise, not edge)."""
    summary = summarize(trades, group_by="strategy")
    if summary.empty:
        return summary
    summary["rank_score"] = summary["expectancy"] * np.log1p(summary["num_trades"])
    return summary.sort_values("rank_score", ascending=False).drop(columns="rank_score").reset_index(drop=True)
