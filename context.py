#!/usr/bin/env python3
"""Market context: exchange/region metadata, weekly trend, index regime, liquidity."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# region -> session timezone, minute the daily bar is final, benchmark index,
# and traded-value thresholds (local currency/day) for the liquidity flag.
# LSE quotes in pence, so no liquidity thresholds are applied outside IN/US.
REGIONS: dict[str, dict[str, Any]] = {
    "IN": dict(tz="Asia/Kolkata", close=(15, 35), bench="^NSEI", bench_name="Nifty", thin=5e7, illiquid=1e7, fixed=timedelta(hours=5, minutes=30)),
    "US": dict(tz="America/New_York", close=(16, 5), bench="^GSPC", bench_name="S&P 500", thin=3e6, illiquid=5e5, fixed=timedelta(hours=-5)),
    "GB": dict(tz="Europe/London", close=(16, 35), bench="^FTSE", bench_name="FTSE 100", thin=None, illiquid=None, fixed=timedelta(0)),
    "JP": dict(tz="Asia/Tokyo", close=(15, 35), bench="^N225", bench_name="Nikkei 225", thin=None, illiquid=None, fixed=timedelta(hours=9)),
    "HK": dict(tz="Asia/Hong_Kong", close=(16, 15), bench="^HSI", bench_name="Hang Seng", thin=None, illiquid=None, fixed=timedelta(hours=8)),
    "CA": dict(tz="America/Toronto", close=(16, 5), bench="^GSPTSE", bench_name="S&P/TSX", thin=None, illiquid=None, fixed=timedelta(hours=-5)),
    "AU": dict(tz="Australia/Sydney", close=(16, 15), bench="^AXJO", bench_name="ASX 200", thin=None, illiquid=None, fixed=timedelta(hours=10)),
    "DE": dict(tz="Europe/Berlin", close=(17, 35), bench="^GDAXI", bench_name="DAX", thin=None, illiquid=None, fixed=timedelta(hours=1)),
    "FR": dict(tz="Europe/Paris", close=(17, 35), bench="^FCHI", bench_name="CAC 40", thin=None, illiquid=None, fixed=timedelta(hours=1)),
    "SG": dict(tz="Asia/Singapore", close=(17, 5), bench="^STI", bench_name="STI", thin=None, illiquid=None, fixed=timedelta(hours=8)),
}
_SUFFIX = {
    ".NS": "IN", ".BO": "IN", ".L": "GB", ".T": "JP", ".HK": "HK",
    ".TO": "CA", ".V": "CA", ".AX": "AU", ".DE": "DE", ".PA": "FR", ".SI": "SG",
}
_INDEX = {
    "^NSEI": "IN", "^NSEBANK": "IN", "^BSESN": "IN", "^CNXIT": "IN",
    "^GSPC": "US", "^IXIC": "US", "^DJI": "US", "^RUT": "US",
    "^FTSE": "GB", "^N225": "JP", "^HSI": "HK", "^GSPTSE": "CA",
    "^AXJO": "AU", "^GDAXI": "DE", "^FCHI": "FR", "^STI": "SG",
}


def region_of(ticker: str) -> str:
    t = (ticker or "").upper()
    if t.startswith("^"):
        return _INDEX.get(t, "US")
    if "." in t:
        return _SUFFIX.get(t[t.rfind(".") :], "US")  # BRK.B etc. fall through to US
    return "US"


def benchmark_for(ticker: str) -> str:
    return REGIONS[region_of(ticker)]["bench"]


def benchmark_name(ticker: str) -> str:
    return REGIONS[region_of(ticker)]["bench_name"]


def _tz(region: str):
    info = REGIONS[region]
    if ZoneInfo is not None:
        try:
            return ZoneInfo(info["tz"])
        except Exception:
            pass
    return timezone(info["fixed"])  # no tzdata (e.g. Windows): fixed offset


def market_now(region: str) -> datetime:
    return datetime.now(_tz(region))


def market_today(region: str) -> date:
    return market_now(region).date()


def session_is_open_for_bar(last_ts: pd.Timestamp, region: str, now: Optional[datetime] = None) -> bool:
    """True if `last_ts` is today's bar in the exchange's own timezone and the
    session has not finished, i.e. the bar is still forming."""
    tz = _tz(region)
    now = now.astimezone(tz) if now is not None else market_now(region)
    ts = pd.Timestamp(last_ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(tz)
    if ts.date() != now.date():
        return False
    h, m = REGIONS[region]["close"]
    return (now.hour, now.minute) < (h, m)


# --------------------------------------------------------------------------
# Weekly confluence
# --------------------------------------------------------------------------
def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def weekly_context(df: pd.DataFrame) -> dict[str, Any]:
    """Higher-timeframe trend. A daily pullback inside a rising weekly trend is
    a very different trade from a daily bounce inside a falling one."""
    out: dict[str, Any] = {"trend": "unknown", "rsi": None, "above_30w": None, "sma30_slope": None}
    try:
        wc = df["Close"].resample("W-FRI").last().dropna()
    except Exception:
        return out
    if len(wc) < 34:
        return out
    sma10 = wc.rolling(10).mean()
    sma30 = wc.rolling(30).mean()
    delta = wc.diff()
    ag = _wilder(delta.clip(lower=0), 14)
    al = _wilder((-delta).clip(lower=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + ag / al)
    c, s30, s10 = float(wc.iloc[-1]), float(sma30.iloc[-1]), float(sma10.iloc[-1])
    slope = float((sma30.iloc[-1] / sma30.iloc[-5] - 1) * 100) if sma30.iloc[-5] else 0.0
    if c > s30 and slope > 0:
        trend = "up"
    elif c < s30 and slope < 0:
        trend = "down"
    else:
        trend = "mixed"
    r = rsi.iloc[-1]
    out.update(
        trend=trend,
        rsi=round(float(r), 1) if np.isfinite(r) else None,
        above_30w=bool(c > s30),
        sma30_slope=round(slope, 2),
        sma10_above_30=bool(s10 > s30),
    )
    return out


# --------------------------------------------------------------------------
# Index regime
# --------------------------------------------------------------------------
def market_regime(bench: Optional[pd.DataFrame], symbol: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"state": "unknown", "symbol": symbol, "drawdown_pct": None}
    if bench is None or len(bench) < 60:
        return out
    c = bench["Close"].astype(float)
    last = float(c.iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    sma200 = c.rolling(200).mean().iloc[-1]
    above50 = last > sma50
    above200 = bool(last > sma200) if np.isfinite(sma200) else above50
    peak = float(c.rolling(252, min_periods=60).max().iloc[-1])
    dd = (last / peak - 1) * 100 if peak else 0.0
    if above200 and above50:
        state = "bull"
    elif (not above200 and not above50) or dd <= -15:
        state = "bear"
    else:
        state = "neutral"
    out.update(state=state, above_200=above200, above_50=bool(above50), drawdown_pct=round(dd, 1))
    return out


# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------
def liquidity_info(df: pd.DataFrame, region: str) -> dict[str, Any]:
    tv = (df["Close"] * df["Volume"]).tail(20)
    adv = float(tv.median()) if len(tv) else 0.0
    zero_range_days = int((df["High"] == df["Low"]).tail(10).sum())
    cfg = REGIONS[region]
    state = "ok"
    if cfg["thin"] is not None:
        if adv < cfg["illiquid"]:
            state = "illiquid"
        elif adv < cfg["thin"]:
            state = "thin"
    if zero_range_days >= 3:
        state = "circuit_or_illiquid"
    return {"state": state, "adv_value": round(adv, 0), "zero_range_days": zero_range_days}
