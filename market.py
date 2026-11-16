#!/usr/bin/env python3
"""Yahoo Finance fetch + short TTL cache."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from analysis import (
    analyze_frame,
    clean_ohlcv,
    empty_error,
    nse_candidate,
    resolve_ticker,
)
from context import benchmark_for

_YF_LOCK = threading.Lock()

CACHE_TTL = 15 * 60
FAIL_TTL = 60  # an empty/failed download is retried after a minute, not 15
MIN_BARS_LISTED = 80

_price_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_info_cache: dict[str, tuple[float, dict]] = {}


def _fresh(entry, ttl=CACHE_TTL):
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > ttl:
        return None
    return payload


def _stamp(ok: bool, ttl: int = CACHE_TTL) -> float:
    """Timestamp that makes an entry expire after FAIL_TTL when it is a failure."""
    return time.time() if ok else time.time() - (ttl - FAIL_TTL)


def download_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    key = (ticker, period)
    cached = _fresh(_price_cache.get(key))
    if cached is not None:
        return cached.copy()
    with _YF_LOCK:
        cached = _fresh(_price_cache.get(key))
        if cached is not None:
            return cached.copy()
        try:
            raw = yf.download(
                ticker, period=period, progress=False, auto_adjust=True, threads=False
            )
            df = clean_ohlcv(raw, ticker=ticker)
        except Exception:
            df = pd.DataFrame()
        _price_cache[key] = (_stamp(len(df) > 0), df)
        return df.copy()


def fetch_info(ticker: str) -> tuple[dict, Any]:
    cached = _fresh(_info_cache.get(ticker), ttl=6 * 3600)
    if cached is not None:
        return cached.get("info") or {}, cached.get("calendar")
    with _YF_LOCK:
        cached = _fresh(_info_cache.get(ticker), ttl=6 * 3600)
        if cached is not None:
            return cached.get("info") or {}, cached.get("calendar")
        t = yf.Ticker(ticker)
        info = {}
        calendar = None
        try:
            info = t.info or {}
        except Exception:
            info = {}
        try:
            calendar = t.calendar
        except Exception:
            calendar = None
        ok = bool(info)
        _info_cache[ticker] = (_stamp(ok, 6 * 3600), {"info": info, "calendar": calendar})
        return info, calendar


def get_benchmark(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Index matched to the ticker's exchange (Nifty for .NS/.BO, S&P 500 for
    US listings, ...). Relative strength vs the wrong index is meaningless."""
    df = download_history(benchmark_for(ticker), period=period)
    return df if len(df) else None


def resolve_listed_ticker(raw: str) -> str:
    ticker = resolve_ticker(raw)
    # Bare symbols in this project are usually NSE. Try .NS first, then the
    # raw Yahoo symbol (US / other listings). Explicit suffixes win as-is.
    candidates = [ticker] if "." in ticker else [nse_candidate(ticker), ticker]
    seen: list[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.append(cand)
        df = download_history(cand)
        if len(df) >= MIN_BARS_LISTED:
            return cand
    raise ValueError(f"No price history for {raw}")


def analyze_ticker(raw: str) -> dict[str, Any]:
    try:
        ticker = resolve_listed_ticker(raw)
    except Exception as exc:
        return empty_error(raw.upper(), str(exc))
    try:
        df = download_history(ticker)
        if df.empty:
            return empty_error(ticker, "No price data")
        info, calendar = fetch_info(ticker)
        bench = get_benchmark(ticker)
        return analyze_frame(ticker, df, bench=bench, info=info, calendar=calendar)
    except Exception as exc:
        return empty_error(ticker, f"{type(exc).__name__}: {exc}")


def invalidate_cache(ticker: Optional[str] = None) -> None:
    if ticker:
        for key in [k for k in _price_cache if k[0] == ticker]:
            _price_cache.pop(key, None)
        _info_cache.pop(ticker, None)
    else:
        _price_cache.clear()
        _info_cache.clear()
