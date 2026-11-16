#!/usr/bin/env python3
"""Walk-forward validation of the buy-zone scores.

For every sampled date the analysis is run on data up to that date ONLY (no
lookahead), then the next bar's open is used as the entry and the outcome is
measured forward: N-day returns, worst drawdown (MAE), whether the invalidation
level or first target was hit first, and the result in R multiples.

    python backtest.py RELIANCE TCS HDFCBANK --period 5y

Caveats: a hand-picked watchlist has survivorship bias, costs/slippage are not
modelled, and outcomes need a decent sample (n >= 30) before they mean much.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis import BUY_ZONES, analyze_frame, clean_ohlcv

HORIZONS = (10, 20, 40)
WARMUP = 260  # bars needed before the first evaluation (200-day average + slope)


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _outcome(o, h, l, c, t: int, entry: float, inv: Optional[float], tgt: Optional[float], horizon: int):
    """First-touch outcome over bars t+1..t+horizon. Stop wins a same-bar tie
    (conservative); gaps through a level fill at the open, not the level."""
    n = len(c)
    end = min(t + horizon, n - 1)
    if inv is None or entry <= inv:
        return None, None
    risk = entry - inv
    for k in range(t + 1, end + 1):
        if l[k] <= inv:
            fill = min(o[k], inv)
            return "stop", (fill - entry) / risk
        if tgt is not None and h[k] >= tgt:
            fill = max(o[k], tgt)
            return "target", (fill - entry) / risk
    if end - t < horizon:
        return None, None  # not enough forward data to call it
    return "timeout", (c[end] - entry) / risk


def walk_forward(
    ticker: str,
    df: pd.DataFrame,
    bench: Optional[pd.DataFrame] = None,
    *,
    warmup: int = WARMUP,
    step: int = 3,
    horizons: tuple[int, ...] = HORIZONS,
    outcome_horizon: int = 40,
) -> pd.DataFrame:
    df = clean_ohlcv(df, ticker=ticker, drop_partial=False)
    n = len(df)
    if n < warmup + max(horizons) + 5:
        raise ValueError(f"{ticker}: need at least {warmup + max(horizons) + 5} bars, have {n}")
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)

    rows: list[dict[str, Any]] = []
    for t in range(warmup, n - 1, step):
        sl = df.iloc[: t + 1]
        b = None
        if bench is not None:
            b = bench[bench.index <= sl.index[-1]]
        try:
            a = analyze_frame(
                ticker, sl, bench=b, drop_partial=False, include_chart=False, now=sl.index[-1].date()
            )
        except Exception:
            continue
        entry = float(o[t + 1])
        if not np.isfinite(entry) or entry <= 0:
            continue
        tg = a["targets"]
        row: dict[str, Any] = {
            "ticker": ticker,
            "date": sl.index[-1],
            "score": a["buy_score"],
            "zone": a["zone"],
            "setup": a["setup"],
            "rr": a["risk_reward"],
            "entry": entry,
            "invalidation": a["invalidation"],
            "target_1": tg["target_1"],
            "in_band": bool(a["buy_from"] and a["buy_from"] <= entry <= (a["buy_to"] or 0) * 1.0),
        }
        for hz in horizons:
            row[f"ret_{hz}"] = (c[t + hz] / entry - 1) * 100 if t + hz <= n - 1 else np.nan
        end = min(t + 20, n - 1)
        row["mae_20"] = (l[t + 1 : end + 1].min() / entry - 1) * 100 if end > t else np.nan
        kind, r_mult = _outcome(o, h, l, c, t, entry, a["invalidation"], tg["target_1"], outcome_horizon)
        row["outcome"] = kind
        row["r"] = r_mult
        rows.append(row)
    return pd.DataFrame(rows)


def dedupe_signals(sig: pd.DataFrame, cooldown_bars: int, step: int) -> pd.DataFrame:
    """Keep one signal per ticker per cooldown window, so a single multi-week
    dip is not counted as 10 independent wins."""
    if sig.empty:
        return sig
    keep = []
    for _, g in sig.sort_values("date").groupby("ticker"):
        last_i = None
        for i, (idx, _row) in enumerate(g.iterrows()):
            if last_i is None or (i - last_i) * step >= cooldown_bars:
                keep.append(idx)
                last_i = i
    return sig.loc[keep]


def _summ(g: pd.DataFrame, base: dict[str, float]) -> dict[str, Any]:
    n = len(g)
    r20 = g["ret_20"].dropna()
    wins = int((r20 > 0).sum())
    lo, hi = _wilson(wins, len(r20))
    out = {
        "n": n,
        "win_rate_20d": round(wins / len(r20) * 100, 1) if len(r20) else None,
        "win_ci95": [round(lo * 100, 1), round(hi * 100, 1)] if len(r20) else None,
        "mean_ret_10d": round(g["ret_10"].mean(), 2),
        "mean_ret_20d": round(r20.mean(), 2) if len(r20) else None,
        "median_ret_20d": round(r20.median(), 2) if len(r20) else None,
        "mean_ret_40d": round(g["ret_40"].mean(), 2),
        "edge_vs_all_20d": round(r20.mean() - base["mean_ret_20d"], 2) if len(r20) else None,
        "avg_mae_20d": round(g["mae_20"].mean(), 2),
        "low_sample": n < 30,
    }
    dec = g.dropna(subset=["outcome"])
    if len(dec):
        out["stop_rate"] = round((dec["outcome"] == "stop").mean() * 100, 1)
        out["target_rate"] = round((dec["outcome"] == "target").mean() * 100, 1)
        out["mean_R"] = round(dec["r"].mean(), 2)
    return out


def summarize(
    results: pd.DataFrame, cooldown_bars: int = 10, step: int = 3
) -> dict[str, Any]:
    if results.empty:
        return {"samples": 0}
    base = {"mean_ret_20d": float(results["ret_20"].mean())}
    summary: dict[str, Any] = {
        "samples": int(len(results)),
        "baseline_all_dates": _summ(results, base),
        "by_zone": {},
        "by_setup": {},
        "by_score_bucket": {},
    }
    for zname, g in results.groupby("zone"):
        summary["by_zone"][zname] = _summ(g, base)
    for sname, g in results.groupby("setup"):
        summary["by_setup"][sname] = _summ(g, base)
    buckets = pd.cut(results["score"], [0, 40, 50, 62, 75, 101], right=False, labels=["<40", "40-50", "50-62", "62-75", "75+"])
    for bname, g in results.groupby(buckets, observed=True):
        summary["by_score_bucket"][str(bname)] = _summ(g, base)
    sig = dedupe_signals(results[results["zone"].isin(BUY_ZONES)], cooldown_bars, step)
    summary["buy_signals_deduped"] = _summ(sig, base) if len(sig) else {"n": 0}
    rr_ok = sig[sig["rr"].fillna(0) >= 1.5]
    summary["buy_signals_rr_ge_1_5"] = _summ(rr_ok, base) if len(rr_ok) else {"n": 0}
    return summary


def run_ticker(raw: str, period: str = "5y", step: int = 3) -> dict[str, Any]:
    """Backtest one live ticker (needs network). Used by the API and CLI."""
    from market import download_history, get_benchmark, resolve_listed_ticker  # lazy: avoids yfinance for unit tests

    ticker = resolve_listed_ticker(raw)
    df = download_history(ticker, period=period)
    bench = get_benchmark(ticker, period=period)
    res = walk_forward(ticker, df, bench=bench, step=step)
    return {"ticker": ticker, "period": period, "summary": summarize(res, step=step)}


def _print(summary: dict[str, Any]) -> None:
    def table(title: str, d: dict[str, dict]):
        if not d:
            return
        print(f"\n== {title}")
        cols = ["n", "win_rate_20d", "mean_ret_20d", "edge_vs_all_20d", "avg_mae_20d", "stop_rate", "target_rate", "mean_R"]
        print(f"{'':22}" + "".join(f"{c:>16}" for c in cols))
        for k, v in d.items():
            cells = "".join(f"{str(v.get(c, '-')):>16}" for c in cols)
            flag = "  (low n)" if v.get("low_sample") else ""
            print(f"{str(k):22}{cells}{flag}")

    print(f"samples: {summary.get('samples')}")
    table("baseline (every sampled date)", {"all": summary.get("baseline_all_dates", {})})
    table("by zone", summary.get("by_zone", {}))
    table("by setup", summary.get("by_setup", {}))
    table("by score bucket", summary.get("by_score_bucket", {}))
    table("buy signals (deduped)", {"buy": summary.get("buy_signals_deduped", {}), "buy & R:R>=1.5": summary.get("buy_signals_rr_ge_1_5", {})})


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward test of the buy-zone score")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--step", type=int, default=3, help="evaluate every N bars")
    args = ap.parse_args()

    from market import download_history, get_benchmark, resolve_listed_ticker

    frames = []
    for raw in args.tickers:
        try:
            t = resolve_listed_ticker(raw)
            df = download_history(t, period=args.period)
            res = walk_forward(t, df, bench=get_benchmark(t, period=args.period), step=args.step)
            frames.append(res)
            print(f"{t}: {len(res)} samples")
        except Exception as exc:
            print(f"{raw}: skipped ({exc})")
    if not frames:
        raise SystemExit("nothing to summarise")
    _print(summarize(pd.concat(frames, ignore_index=True), step=args.step))


if __name__ == "__main__":
    main()
