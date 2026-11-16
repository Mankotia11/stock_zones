#!/usr/bin/env python3
"""Technical + context analysis for the stock tracker dashboard.

v2 changes (see UPGRADE_NOTES.md): zone-based support/resistance, ATR-scaled
bands and stops, R:R gating, weekly + index-regime confluence, liquidity flag,
candle confirmation, quality-filtered divergence, per-exchange benchmark and
session handling.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from context import (
    benchmark_for,
    benchmark_name,
    liquidity_info,
    market_regime,
    market_today,
    region_of,
    session_is_open_for_bar,
    weekly_context,
)
from zones import (
    ZIGZAG_ATR_MULT,
    atr_last,
    breakout_retest,
    candle_signals,
    find_zones,
    pick_resistances,
    pick_support,
    pivot_idx as _pivot_idx,
    recent_breakdown,
    zigzag,
)

try:
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # no tzdata (e.g. Windows) -> fixed offset, India has no DST
    IST = timezone(timedelta(hours=5, minutes=30))

MIN_BARS = 80  # minimum to analyse at all
FULL_HISTORY_BARS = 220  # needed for SMA200 + its slope
BENCHMARK = "^NSEI"  # default (Indian) benchmark; see context.benchmark_for()
PIVOT_ORDER = 3
RECENT_PIVOT_BARS = 90  # latest zig-zag pivot must be this fresh to define structure
STALE_DAYS = 10  # last bar older than this (calendar days) -> data is stale

OHLCV_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

BUY_ZONES = ("strong_buy_zone", "buy_zone")


# --------------------------------------------------------------------------
# Ticker helpers
# --------------------------------------------------------------------------
def resolve_ticker(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "")
    if t.startswith("$"):
        t = t[1:]
    if not t:
        raise ValueError("Empty ticker")
    return t


def nse_candidate(ticker: str) -> str:
    """Append .NS for bare NSE symbols; leave indices (^NSEI), FX (X=) and suffixed tickers alone."""
    if "." in ticker or ticker.startswith("^") or "=" in ticker:
        return ticker
    return f"{ticker}.NS"


# --------------------------------------------------------------------------
# Data cleaning
# --------------------------------------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yfinance MultiIndex frame. Only safe for a single ticker."""
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df

    field_level = None
    for lvl in range(df.columns.nlevels):
        if set(df.columns.get_level_values(lvl)) & OHLCV_FIELDS:
            field_level = lvl
            break
    if field_level is None:
        return df

    for lvl in range(df.columns.nlevels):
        if lvl == field_level:
            continue
        others = {str(x) for x in df.columns.get_level_values(lvl) if str(x) != ""}
        if len(others) > 1:
            raise ValueError(
                "Multi-ticker frame passed to flatten_columns; analyse one ticker at a time"
            )

    result = df.copy()
    result.columns = df.columns.get_level_values(field_level)
    return result.loc[:, ~result.columns.duplicated()]


def clean_ohlcv(df: pd.DataFrame, ticker: Optional[str] = None, drop_partial: bool = True) -> pd.DataFrame:
    """Normalise a yfinance frame.

    drop_partial removes today's bar while that exchange's session is still
    open (it is incomplete). The session is looked up per ticker (NSE/BSE,
    NYSE/Nasdaq, LSE, ...); with no ticker the legacy NSE rule applies.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = flatten_columns(df.copy())
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]  # yfinance sometimes repeats the last bar
    # Rare bad prints: missing Open, or O/C outside the High-Low range.
    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    if drop_partial and len(df):
        region = region_of(ticker) if ticker else "IN"
        if session_is_open_for_bar(df.index[-1], region):
            df = df.iloc[:-1]
    return df


def _naive_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with a tz-naive, midnight-normalised index so two series from
    different venues/tz conventions can be aligned by calendar date."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")]


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _safe(val, default=None):
    try:
        if val is None:
            return default
        x = float(val)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def swing_points(series: pd.Series, order: int = 3) -> list[int]:
    """Backward-compatible helper: all confirmed fractal swing highs and lows, sorted."""
    idx = np.union1d(_pivot_idx(series, order, "low"), _pivot_idx(series, order, "high"))
    return [int(i) for i in idx]


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI: no losses -> 100 (not NaN); no movement at all -> 50.
    delta = c.diff()
    avg_gain = _wilder(delta.clip(lower=0), 14)
    avg_loss = _wilder((-delta).clip(lower=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df["RSI"] = rsi

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = _wilder(tr, 14)
    df["ATR_pct"] = df["ATR"] / c * 100

    up = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_dm = up.where(up > down, 0.0)
    minus_dm = down.where(down > up, 0.0)
    atr14 = df["ATR"].replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, 14) / atr14
    minus_di = 100 * _wilder(minus_dm, 14) / atr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    df["ADX"] = _wilder(dx, 14)
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    mid = df["SMA20"]
    std = c.rolling(20).std()
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid.replace(0, np.nan)
    df["BB_pct"] = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # Relative volume vs the *prior* 20 sessions (current bar excluded).
    df["Vol_SMA20"] = v.rolling(20).mean()
    df["Rel_Vol"] = v / df["Vol_SMA20"].shift(1).replace(0, np.nan)
    obv = np.sign(c.diff().fillna(0)) * v
    df["OBV"] = obv.cumsum()
    df["OBV_slope"] = df["OBV"].diff(10)  # OBV is signed; pct_change is meaningless

    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm.fillna(0) * v
    df["CMF"] = mfv.rolling(21).sum() / v.rolling(21).sum().replace(0, np.nan)

    df["ROC_20"] = c.pct_change(20) * 100
    df["ROC_60"] = c.pct_change(60) * 100
    roll_max_60 = c.rolling(60, min_periods=20).max()
    roll_max_252 = c.rolling(252, min_periods=60).max()
    roll_min_252 = c.rolling(252, min_periods=60).min()
    df["DD_20"] = (c / c.rolling(20, min_periods=5).max() - 1) * 100
    df["DD_60"] = (c / roll_max_60 - 1) * 100
    df["Pos_52w"] = (c - roll_min_252) / (roll_max_252 - roll_min_252).replace(0, np.nan)

    df["SMA20_slope"] = df["SMA20"].pct_change(10) * 100
    df["SMA50_slope"] = df["SMA50"].pct_change(10) * 100
    df["SMA200_slope"] = df["SMA200"].pct_change(20) * 100
    return df


# --------------------------------------------------------------------------
# Structure + fibonacci
# --------------------------------------------------------------------------
def _fib_zone(
    price: float,
    swing_low: Optional[float],
    swing_high: Optional[float],
    low_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
) -> dict[str, Any]:
    empty = {"in_zone": False, "level": None, "retracement": None}
    if not swing_high or not swing_low or swing_high <= swing_low:
        return empty
    # A pullback retracement only makes sense when the low came BEFORE the high.
    if low_idx is not None and high_idx is not None and low_idx >= high_idx:
        return empty

    span = swing_high - swing_low
    retr = (swing_high - price) / span
    levels = {
        "0.382": swing_high - 0.382 * span,
        "0.5": swing_high - 0.5 * span,
        "0.618": swing_high - 0.618 * span,
    }
    in_zone = 0.35 <= retr <= 0.68
    nearest = min(levels.items(), key=lambda kv: abs(price - kv[1]))
    return {
        "in_zone": in_zone,
        "level": nearest[0],
        "retracement": round(float(retr), 3),
        "prices": {k: round(v, 2) for k, v in levels.items()},
    }


def _structure(df: pd.DataFrame, price: float, atr: float) -> dict[str, Any]:
    """Market structure from ATR zig-zag swings (not 3-bar fractals).

    Only moves large enough to matter (>= ZIGZAG_ATR_MULT ATR) count as swings,
    which removes the noise that made HH/HL/LH/LL labels flip on tiny wiggles.
    A live break of the last swing counts, so structure does not lag.
    """
    n = len(df)
    piv = zigzag(
        df["High"].to_numpy(dtype=float),
        df["Low"].to_numpy(dtype=float),
        df["Close"].to_numpy(dtype=float),
        df["ATR"].to_numpy(dtype=float),
    )
    highs = [p for p in piv if p["t"] == "H"]
    lows = [p for p in piv if p["t"] == "L"]

    def fresh(p) -> bool:
        return (n - 1 - p["i"]) <= RECENT_PIVOT_BARS

    hh = len(highs) >= 2 and fresh(highs[-1]) and highs[-1]["p"] > highs[-2]["p"]
    lh = len(highs) >= 2 and fresh(highs[-1]) and highs[-1]["p"] < highs[-2]["p"]
    hl = len(lows) >= 2 and fresh(lows[-1]) and lows[-1]["p"] > lows[-2]["p"]
    ll = len(lows) >= 2 and fresh(lows[-1]) and lows[-1]["p"] < lows[-2]["p"]
    if len(highs) >= 2 and fresh(highs[-1]) and price > max(highs[-1]["p"], highs[-2]["p"]):
        hh, lh = True, False  # live break above both prior swing highs
    if len(lows) >= 2 and fresh(lows[-1]) and price < min(lows[-1]["p"], lows[-2]["p"]):
        ll, hl = True, False

    rsi = df["RSI"].to_numpy(dtype=float)
    atr_arr = df["ATR"].to_numpy(dtype=float)

    # Bullish divergence: a real lower low (>= 0.25 ATR) with RSI >= 2 pts higher,
    # swings 5-90 bars apart, still intact (price has not undercut the 2nd low).
    rsi_div = False
    if len(lows) >= 2 and (n - 1 - lows[-1]["i"]) <= 40:
        a, b = lows[-2], lows[-1]
        gap = b["i"] - a["i"]
        a_atr = atr_arr[b["i"]] if np.isfinite(atr_arr[b["i"]]) else atr
        if (
            5 <= gap <= 90
            and b["p"] < a["p"] - 0.25 * a_atr
            and np.isfinite(rsi[a["i"]])
            and np.isfinite(rsi[b["i"]])
            and rsi[b["i"]] > rsi[a["i"]] + 2
            and price > b["p"]
        ):
            rsi_div = True

    # Bearish divergence: higher high with RSI lower (distribution warning).
    rsi_bear_div = False
    if len(highs) >= 2 and (n - 1 - highs[-1]["i"]) <= 40:
        a, b = highs[-2], highs[-1]
        gap = b["i"] - a["i"]
        a_atr = atr_arr[b["i"]] if np.isfinite(atr_arr[b["i"]]) else atr
        if (
            5 <= gap <= 90
            and b["p"] > a["p"] + 0.25 * a_atr
            and np.isfinite(rsi[a["i"]])
            and np.isfinite(rsi[b["i"]])
            and rsi[b["i"]] < rsi[a["i"]] - 2
        ):
            rsi_bear_div = True

    tail = df.iloc[-60:]
    last_high = float(highs[-1]["p"]) if highs else float(tail["High"].max())
    last_low = float(lows[-1]["p"]) if lows else float(tail["Low"].min())

    # Most recent completed up-leg for retracement maths: last swing high and
    # the swing low just before it. Must span >= 3 ATR to be a meaningful leg.
    leg = None
    if highs:
        H = highs[-1]
        prior = [p for p in lows if p["i"] < H["i"]]
        if prior and (H["p"] - prior[-1]["p"]) >= 3 * atr:
            leg = (prior[-1]["p"], H["p"], prior[-1]["i"], H["i"])

    label = "range"
    if hh and hl:
        label = "uptrend"
    elif lh and ll:
        label = "downtrend"
    elif hl and lh:
        label = "contracting"  # higher lows + lower highs: coiling range
    elif ll and hh:
        label = "expanding"  # lower lows + higher highs: widening range

    return {
        "hh": bool(hh),
        "hl": bool(hl),
        "lh": bool(lh),
        "ll": bool(ll),
        "rsi_div": bool(rsi_div),
        "rsi_bear_div": bool(rsi_bear_div),
        "label": label,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
        "leg": leg,
        "n_swings": len(piv),
    }


def _today_ist():
    return datetime.now(IST).date()


def _earnings_soon(calendar, today: Optional[date] = None) -> bool:
    """True if an earnings date falls within the next 10 days."""
    try:
        if calendar is None:
            return False
        today = today or _today_ist()
        dates: list = []
        if isinstance(calendar, pd.DataFrame):
            if calendar.empty:
                return False
            if "Earnings Date" in calendar.columns:
                dates = calendar["Earnings Date"].dropna().tolist()
            elif "Earnings Date" in calendar.index:  # older yfinance: row layout
                row = calendar.loc["Earnings Date"]
                dates = row.dropna().tolist() if hasattr(row, "dropna") else [row]
        elif isinstance(calendar, dict):
            d = calendar.get("Earnings Date", [])
            dates = d if isinstance(d, (list, tuple)) else [d]
        for ed in dates:
            try:
                diff = (pd.Timestamp(ed).date() - today).days
                if 0 <= diff <= 10:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def classify_zone(score: float, setup: str) -> str:
    if setup in ("Falling knife", "Breakdown"):
        return "avoid"
    if setup == "Extended":
        return "extended"
    if score >= 75:
        return "strong_buy_zone"
    if score >= 62:
        return "buy_zone"
    if score >= 50:
        return "watch"
    return "neutral"


def _rr_points(rr: Optional[float]) -> int:
    if rr is None:
        return 35
    if rr >= 3:
        return 90
    if rr >= 2:
        return 75
    if rr >= 1.5:
        return 60
    if rr >= 1:
        return 45
    return 20


def _zone_view(zones: list[dict], price: float, atr: float, n_each: int = 3, span: float = 0.35) -> dict[str, list]:
    keep = [z for z in zones if abs(z["mid"] / price - 1) <= span]
    below = sorted((z for z in keep if z["mid"] <= price), key=lambda z: -z["mid"])[:n_each]
    above = sorted((z for z in keep if z["mid"] > price), key=lambda z: z["mid"])[:n_each]
    return {"support": below, "resistance": above}


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def analyze_frame(
    ticker: str,
    df: pd.DataFrame,
    bench: Optional[pd.DataFrame] = None,
    info: Optional[dict] = None,
    calendar=None,
    *,
    drop_partial: bool = True,
    include_chart: bool = True,
    now: Optional[date] = None,
) -> dict[str, Any]:
    region = region_of(ticker)
    bench_symbol = benchmark_for(ticker)
    df = clean_ohlcv(df, ticker=ticker, drop_partial=drop_partial)
    if len(df) < MIN_BARS:
        raise ValueError(f"{ticker}: need at least {MIN_BARS} daily bars")
    limited_history = len(df) < FULL_HISTORY_BARS
    df = add_indicators(df)
    info = info or {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    price = float(last["Close"])
    today = now or market_today(region)
    stale_days = (today - df.index[-1].date()).days
    stale = stale_days > STALE_DAYS

    atr = _safe(last["ATR"]) or price * 0.02
    atr_pct = atr / price * 100
    near_pct = float(np.clip(1.2 * atr_pct, 1.0, 5.0))  # "near a MA" scales with volatility

    rsi = _safe(last["RSI"], 50.0)
    rsi_prev = _safe(prev["RSI"], rsi)
    rsi_prev2 = _safe(prev2["RSI"], rsi_prev)
    macd_hist = _safe(last["MACD_hist"], 0.0)
    macd_hist_prev = _safe(prev["MACD_hist"], 0.0)
    macd_hist_prev2 = _safe(prev2["MACD_hist"], macd_hist_prev)

    struct = _structure(df, price, atr)
    fib = (
        _fib_zone(price, *struct["leg"][:2], struct["leg"][2], struct["leg"][3])
        if struct["leg"]
        else {"in_zone": False, "level": None, "retracement": None}
    )

    # ---- zones: support, resistance, failed support, breakout retest -------
    zones = find_zones(df)
    res_list = pick_resistances(zones, price, atr)
    sup = pick_support(zones, price, atr)
    breakdown = recent_breakdown(df, zones, atr)
    retest = breakout_retest(df, zones, atr)
    if retest is not None:
        sup = retest["zone"]  # old resistance now acting as support
    support_failed = breakdown is not None and (sup is None or breakdown["mid"] > sup["mid"])
    candles = candle_signals(df, sup, atr)
    candle_confirm = candles["rejection"] or candles["engulfing"] or candles["reclaim"]

    sup_gap_atr = None
    at_support = False
    if sup is not None:
        sup_gap_atr = max(0.0, price - sup["high"]) / atr
        at_support = (
            price >= sup["low"] - 0.5 * atr
            and price <= sup["high"] + 1.2 * atr
            and not support_failed
        )
    supply_close = any(
        z is not sup and z["strength"] >= 35 and z["mid"] > price and z["low"] - price <= 0.6 * atr
        for z in zones
    )
    fib_confluence = False
    if fib["in_zone"] and sup is not None and fib.get("prices"):
        fib_confluence = any(
            sup["low"] - 0.5 * atr <= fib["prices"][k] <= sup["high"] + 0.5 * atr for k in ("0.5", "0.618")
        )

    # ---- moving averages ----------------------------------------------------
    sma20 = _safe(last["SMA20"])
    sma50 = _safe(last["SMA50"])
    sma200 = _safe(last["SMA200"])
    has_200 = sma200 is not None
    above_20 = sma20 is not None and price > sma20
    above_50 = sma50 is not None and price > sma50
    above_200 = has_200 and price > sma200
    # With <200 bars, SMA200 is unknown (not "bearish"): fall back to the 50-day.
    long_trend_ok = above_200 if has_200 else above_50
    ma_stack_bull = (
        sma20 is not None and sma50 is not None and has_200 and sma20 > sma50 > sma200
    )

    dist_20 = ((price / sma20) - 1) * 100 if sma20 else None
    dist_50 = ((price / sma50) - 1) * 100 if sma50 else None
    dist_200 = ((price / sma200) - 1) * 100 if sma200 else None
    ext_atr = (price - sma20) / atr if sma20 else 0.0  # stretch above the 20-day, in ATRs
    dist_support = ((price / sup["mid"]) - 1) * 100 if sup else None

    rel_vol = _safe(last["Rel_Vol"], 1.0)
    cmf = _safe(last["CMF"], 0.0)
    adx = _safe(last["ADX"], 15.0)
    plus_di = _safe(last["PLUS_DI"], 20.0)
    minus_di = _safe(last["MINUS_DI"], 20.0)
    bb_pct = _safe(last["BB_pct"], 0.5)
    bb_width = _safe(last["BB_width"], 0.1)
    bb_width_med = _safe(df["BB_width"].rolling(120).median().iloc[-1], bb_width)
    squeeze = bb_width < 0.75 * bb_width_med
    stoch_k = _safe(last["STOCH_K"], 50.0)
    stoch_d = _safe(last["STOCH_D"], 50.0)
    stoch_prev_k = _safe(prev["STOCH_K"], stoch_k)
    stoch_prev_d = _safe(prev["STOCH_D"], stoch_d)
    roc20 = _safe(last["ROC_20"], 0.0)
    dd60 = _safe(last["DD_60"], 0.0)
    pos52 = _safe(last["Pos_52w"], 0.5)

    # ---- relative strength + regime (benchmark matched to the exchange) -----
    rs_20: Optional[float] = None
    rs_60: Optional[float] = None
    rs_line_up: Optional[bool] = None
    regime = {"state": "unknown", "symbol": bench_symbol, "drawdown_pct": None}
    if bench is not None:
        b = clean_ohlcv(bench, ticker=bench_symbol, drop_partial=drop_partial)
        if len(b):
            regime = market_regime(b, bench_symbol)
        if len(b) >= 25:
            sn, bn = _naive_daily(df), _naive_daily(b)
            common = sn.index.intersection(bn.index)
            if len(common) >= 25:
                s = sn.loc[common, "Close"]
                x = bn.loc[common, "Close"]
                rs_20 = float((s.iloc[-1] / s.iloc[-21] - 1) - (x.iloc[-1] / x.iloc[-21] - 1)) * 100
                if len(common) >= 65:
                    rs_60 = float((s.iloc[-1] / s.iloc[-61] - 1) - (x.iloc[-1] / x.iloc[-61] - 1)) * 100
                if len(common) >= 60:
                    ratio = s / x
                    rs_line_up = bool(ratio.iloc[-1] > ratio.rolling(50).mean().iloc[-1])

    weekly = weekly_context(df)
    liq = liquidity_info(df, region)

    # Two-bar confirmation instead of a single-bar uptick.
    rsi_turning_up = rsi > rsi_prev > rsi_prev2 and min(rsi_prev, rsi_prev2) < 45
    macd_turning_up = macd_hist > macd_hist_prev > macd_hist_prev2 and macd_hist_prev < 0
    stoch_cross_up = stoch_k > stoch_d and stoch_prev_k <= stoch_prev_d

    near_ma = any(d is not None and abs(d) <= near_pct for d in (dist_20, dist_50, dist_200))
    oversold_band = 28 <= rsi <= 48
    not_crushed = rsi >= 28 or struct["hl"] or struct["rsi_div"]

    # ---- buy band, invalidation, targets, risk/reward (all ATR-scaled) ------
    buy_from = buy_to = invalidation = entry_ref = target_1 = target_2 = None
    target_src = None
    rr = None
    if sup is not None and not support_failed:
        buy_from = sup["low"]
        buy_to = sup["high"] + 0.35 * atr
        for ma in (sma20, sma50, sma200):  # extend to a MA sitting just above the zone
            if ma and sup["high"] < ma <= sup["high"] + 0.9 * atr:
                buy_to = max(buy_to, ma + 0.1 * atr)
        buy_to = min(buy_to, sup["high"] + 1.0 * atr)
        invalidation = min(sup["low"] - 0.6 * atr, sup["low"] * 0.99)

        if price > invalidation:
            entry_ref = price if price <= buy_to else buy_to
            # First target = next *significant* supply zone at least 1.5 ATR away,
            # so a one-touch bump right above price does not define the reward.
            cands = [z for z in res_list if z["low"] > entry_ref + 1.5 * atr and z["strength"] >= 40]
            if not cands:
                cands = [z for z in res_list if z["low"] > entry_ref + 0.5 * atr]
            if cands:
                target_1 = cands[0]["low"]  # conservative: front edge of supply
                target_2 = cands[1]["low"] if len(cands) > 1 else None
                target_src = "zone"
            else:
                target_1 = entry_ref + 2.5 * atr  # blue sky: no overhead zone found
                target_src = "projected"
            risk = entry_ref - invalidation
            reward = target_1 - entry_ref
            if risk > 0 and reward > 0:
                rr = round(reward / risk, 2)

    # ---- sub-scores -------------------------------------------------------
    location = 40
    if at_support:
        location += 10 + int(14 * sup["strength"] / 100)
    if retest is not None and at_support:
        location += 6
    if near_ma and long_trend_ok:
        location += 18
    elif near_ma:
        location += 10
    if fib["in_zone"]:
        location += 12
    if fib_confluence:
        location += 6
    if bb_pct < 0.25:
        location += 10
    if candle_confirm and at_support:
        location += 10
    if ext_atr > 3:
        location -= 18
    elif ext_atr > 2:
        location -= 8
    if pos52 > 0.92:
        location -= 12
    if sup_gap_atr is not None and sup_gap_atr > 6:
        location -= 8
    if sup is None:
        location -= 10
    if supply_close:
        location -= 10
    if support_failed:
        location -= 25
    location = int(np.clip(location, 0, 100))

    momentum = 40
    if rsi_turning_up:
        momentum += 16
    if macd_turning_up:
        momentum += 14
    if stoch_cross_up and stoch_k < 55:
        momentum += 10
    if struct["rsi_div"]:
        momentum += 16
    if struct["hl"]:
        momentum += 12
    if plus_di > minus_di and adx >= 18:
        momentum += 8
    if candle_confirm and at_support:
        momentum += 8
    if rsi < 25 and not struct["hl"]:
        momentum -= 18
    if macd_hist < macd_hist_prev and macd_hist < 0:
        momentum -= 8
    if struct["rsi_bear_div"]:
        momentum -= 12
    momentum = int(np.clip(momentum, 0, 100))

    trend = 40
    if above_200:
        trend += 18
    if above_50:
        trend += 10
    if ma_stack_bull:
        trend += 14
    if _safe(last["SMA200_slope"], 0) > 0:
        trend += 10
    if rs_20 is not None:
        if rs_20 > 2:
            trend += 6
        elif rs_20 < -4:
            trend -= 12
    if rs_60 is not None and rs_60 > 3:
        trend += 4
    if rs_line_up:
        trend += 4
    if struct["label"] == "uptrend":
        trend += 10
    elif struct["label"] == "downtrend":
        trend -= 16
    if adx < 15:
        trend -= 4
    if weekly["trend"] == "up":
        trend += 10
    elif weekly["trend"] == "down":
        trend -= 12
    trend = int(np.clip(trend, 0, 100))

    volume = 45
    if rel_vol < 0.85 and (near_ma or at_support):
        volume += 10  # quiet pullback
    if rel_vol > 1.4 and (rsi_turning_up or macd_turning_up or candle_confirm):
        volume += 16  # demand returning
    if cmf > 0.05:
        volume += 10
    elif cmf < -0.08:
        volume -= 10
    if _safe(last["OBV_slope"], 0) > 0:
        volume += 8
    if retest is not None and retest["vol_confirm"]:
        volume += 8
    volume = int(np.clip(volume, 0, 100))

    risk_score = _rr_points(rr)
    if sup is not None:
        risk_score += int((sup["strength"] - 50) * 0.2)
    if liq["state"] == "thin":
        risk_score -= 8
    elif liq["state"] in ("illiquid", "circuit_or_illiquid"):
        risk_score -= 20
    if atr_pct > 6:
        risk_score -= 10
    risk_score = int(np.clip(risk_score, 0, 100))

    # ---- setup classification --------------------------------------------
    earnings_risk = _earnings_soon(calendar, today)
    weekly_down = weekly["trend"] == "down"
    falling_knife = (
        not above_50
        and (not has_200 or not above_200)
        and _safe(last["SMA50_slope"], 0) < -2
        and rsi < 32
        and struct["ll"]
        and not struct["hl"]
    )
    extended = (ext_atr > 3 or (dist_20 is not None and dist_20 > 8)) and rsi > 65 and pos52 > 0.85
    quality_pullback = (
        long_trend_ok
        and not weekly_down
        and oversold_band
        and (near_ma or at_support or fib["in_zone"])
        and (struct["hl"] or rsi_turning_up or macd_turning_up or candle_confirm)
        and not falling_knife
        and not support_failed
    )
    breakout_retest_ok = (
        retest is not None
        and long_trend_ok
        and 40 <= rsi <= 68
        and not extended
        and not support_failed
    )
    momentum_dip = (
        above_50
        and abs(dd60) < 12
        and rs_20 is not None
        and rs_20 > 1
        and (near_ma or oversold_band)
        and not extended
        and not support_failed
    )
    reversal = (
        (struct["rsi_div"] or (rsi_turning_up and rsi < 40))
        and not long_trend_ok
        and not_crushed
        and (at_support or candles["reclaim"])  # a reversal needs a level to reverse from
    )
    squeeze_break = squeeze and macd_turning_up and plus_di > minus_di and not support_failed

    setup = "Neutral"
    setup_mult = 1.0
    if falling_knife:
        setup, setup_mult = "Falling knife", 0.45
    elif support_failed:
        setup, setup_mult = "Breakdown", 0.5
    elif extended:
        setup, setup_mult = "Extended", 0.55
    elif quality_pullback:
        setup, setup_mult = "Quality pullback", 1.18
    elif breakout_retest_ok:
        setup, setup_mult = "Breakout retest", 1.15
    elif momentum_dip:
        setup, setup_mult = "Momentum dip", 1.08
    elif squeeze_break:
        setup, setup_mult = "Squeeze expansion", 1.05
    elif reversal:
        setup, setup_mult = "Reversal attempt", 0.92
    elif struct["label"] == "downtrend" and rsi < 40:
        setup, setup_mult = "Breakdown", 0.5

    risk_adj = 1.0
    if earnings_risk:
        risk_adj *= 0.9
    if atr_pct > 5:
        risk_adj *= 0.95
    if regime["state"] == "bear":
        risk_adj *= 0.92
    if liq["state"] == "thin":
        risk_adj *= 0.95
    elif liq["state"] in ("illiquid", "circuit_or_illiquid"):
        risk_adj *= 0.85

    raw = (
        location * 0.28
        + momentum * 0.24
        + trend * 0.20
        + volume * 0.12
        + risk_score * 0.16
    )
    score = round(float(np.clip(raw * setup_mult * risk_adj, 0, 100)), 1)
    zone = classify_zone(score, setup)

    # ---- actionability gates ---------------------------------------------
    gate_note = None
    if zone in BUY_ZONES:
        if buy_from is None:
            zone, gate_note = "watch", "Score is buy-grade but no demand zone defines an entry or stop - capped at Watch"
        elif rr is not None and rr < 1.0:
            zone, gate_note = "watch", f"Score is buy-grade but reward:risk is only {rr:.1f} - capped at Watch"
        elif rr is not None and rr < 1.5 and zone == "strong_buy_zone":
            zone, gate_note = "buy_zone", f"Reward:risk {rr:.1f} is below 1.5 - downgraded from strong buy"
    if stale:
        zone, gate_note = "neutral", f"Last bar is {stale_days} days old - data is stale, no signal"

    if zone in ("avoid", "extended"):
        buy_from = buy_to = invalidation = entry_ref = target_1 = target_2 = None
        rr = None
        target_src = None

    risk_pct = size_pct = None
    if invalidation is not None and entry_ref:
        risk_pct = round((entry_ref - invalidation) / entry_ref * 100, 2)
        if risk_pct > 0:
            size_pct = round(min(100.0, 100.0 / risk_pct), 1)  # % of capital to risk 1% on the stop

    # ---- signals ------------------------------------------------------------
    bname = benchmark_name(ticker)
    signals: list[str] = []
    if quality_pullback:
        signals.append("Healthy trend, price back into a demand area")
    if breakout_retest_ok:
        v = " on strong volume" if retest["vol_confirm"] else ""
        signals.append(f"Broke out{v} {retest['bars_ago']} bars ago, now retesting the old resistance as support")
    if momentum_dip:
        signals.append("Shallow dip while relative strength stays positive")
    if at_support:
        signals.append(
            f"At demand zone {sup['low']:.2f}-{sup['high']:.2f} "
            f"({sup['touches']} touches, strength {sup['strength']})"
        )
    if candle_confirm and at_support:
        kinds = [k for k in ("rejection", "engulfing", "reclaim") if candles[k]]
        signals.append("Candle confirmation at the zone: " + ", ".join(kinds))
    if weekly["trend"] == "up":
        signals.append("Weekly trend is up: daily dip inside a rising higher-timeframe trend")
    elif weekly_down:
        signals.append("Weekly trend is down: bounces are counter-trend")
    if rsi_turning_up:
        signals.append(f"RSI rising two bars running ({rsi_prev2:.0f} -> {rsi:.0f})")
    if macd_turning_up:
        signals.append("MACD histogram improving from below zero")
    if struct["hl"]:
        signals.append("Higher swing low in place")
    if struct["rsi_div"]:
        signals.append("Bullish RSI divergence at the last two swing lows")
    if struct["rsi_bear_div"]:
        signals.append("Bearish RSI divergence at the last two swing highs - momentum fading")
    if fib["in_zone"]:
        extra = " (aligns with the demand zone)" if fib_confluence else ""
        signals.append(f"Trading in the {fib['level']} Fibonacci retracement band{extra}")
    if near_ma and long_trend_ok:
        signals.append("Holding the moving-average cluster in a healthy trend")
    if squeeze:
        signals.append("Bollinger squeeze: volatility compressed, move likely soon")
    if rs_20 is not None and rs_20 > 2:
        signals.append(f"Outperforming {bname} by {rs_20:.1f} pts over 20 sessions")
    if cmf > 0.05:
        signals.append("Chaikin money flow positive (buying pressure)")
    if supply_close:
        signals.append("Strong supply zone directly overhead - limited room")
    if regime["state"] == "bear":
        signals.append(f"{bname} is in a downtrend - broad-market headwind")
    if earnings_risk:
        signals.append("Earnings within 10 days - size down or wait")
    if liq["state"] in ("thin", "illiquid", "circuit_or_illiquid"):
        signals.append(f"Liquidity flag ({liq['state'].replace('_', ' ')}): expect slippage / gap risk")
    if falling_knife:
        signals.append("Declining MAs, lower lows, weak RSI - not a buy zone")
    if support_failed:
        signals.append(f"Support zone {breakdown['low']:.2f}-{breakdown['high']:.2f} just failed - wait for a base")
    if extended:
        signals.append("Stretched above the 20-day and near the top of its 52-week range - chase risk")
    if gate_note:
        signals.append(gate_note)
    if limited_history:
        signals.append(f"Under {FULL_HISTORY_BARS} bars of history: 200-day average unavailable")
    if not signals:
        signals.append("No strong edge: wait for a test of support or a momentum turn")

    change_1d = float(df["Close"].pct_change().iloc[-1] * 100)

    name = info.get("shortName") or info.get("longName") or ticker.replace(".NS", "")
    sector = info.get("sector") or "Unknown"
    pe = _safe(info.get("trailingPE"))
    mcap = _safe(info.get("marketCap"))
    beta = _safe(info.get("beta"))

    zone_view = _zone_view(zones, price, atr)
    resistance = res_list[0]["low"] if res_list else struct["last_swing_high"]

    result = {
        "ticker": ticker,
        "symbol": ticker.replace(".NS", "").replace(".BO", ""),
        "name": name,
        "sector": sector,
        "price": round(price, 2),
        "change_1d": round(change_1d, 2),
        "change_20d": round(roc20, 2),
        "buy_score": score,
        "zone": zone,
        "setup": setup,
        "signals": signals[:10],
        "buy_from": round(buy_from, 2) if buy_from is not None else None,
        "buy_to": round(buy_to, 2) if buy_to is not None else None,
        "invalidation": round(invalidation, 2) if invalidation is not None else None,
        "support": round(sup["mid"], 2) if sup else None,
        "resistance": round(float(resistance), 2),
        "targets": {
            "entry_ref": round(entry_ref, 2) if entry_ref is not None else None,
            "target_1": round(target_1, 2) if target_1 is not None else None,
            "target_2": round(target_2, 2) if target_2 is not None else None,
            "source": target_src,
        },
        "risk_reward": rr,
        "risk_pct": risk_pct,
        "size_pct_for_1pct_risk": size_pct,
        "in_buy_band": bool(buy_from and buy_to and buy_from <= price <= buy_to),
        "limited_history": limited_history,
        "stale_days": stale_days,
        "scores": {
            "location": location,
            "momentum": momentum,
            "trend": trend,
            "volume": volume,
            "risk": risk_score,
        },
        "technicals": {
            "rsi": round(rsi, 1),
            "rsi_prev": round(rsi_prev, 1),
            "macd_hist": round(macd_hist, 4),
            "adx": round(adx, 1),
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "stoch_k": round(stoch_k, 1),
            "stoch_d": round(stoch_d, 1),
            "rel_vol": round(rel_vol, 2),
            "cmf": round(cmf, 3),
            "bb_pct": round(bb_pct, 2),
            "atr": round(atr, 2),
            "atr_pct": round(atr_pct, 2),
            "sma20": round(sma20, 2) if sma20 else None,
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "dist_sma20": round(dist_20, 2) if dist_20 is not None else None,
            "dist_sma50": round(dist_50, 2) if dist_50 is not None else None,
            "dist_sma200": round(dist_200, 2) if dist_200 is not None else None,
            "dd_60": round(dd60, 1),
            "pos_52w": round(pos52 * 100, 1),
            "rs_20": round(rs_20, 2) if rs_20 is not None else None,
            "rs_60": round(rs_60, 2) if rs_60 is not None else None,
            "rs_line_up": rs_line_up,
            "structure": struct["label"],
            "higher_low": struct["hl"],
            "higher_high": struct["hh"],
            "rsi_divergence": struct["rsi_div"],
            "rsi_bear_divergence": struct["rsi_bear_div"],
            "squeeze": bool(squeeze),
            "above_200dma": bool(above_200),
            "ma_stack_bull": bool(ma_stack_bull),
        },
        "fundamentals": {
            "pe": round(pe, 1) if pe else None,
            "market_cap": mcap,
            "beta": round(beta, 2) if beta else None,
            "earnings_soon": earnings_risk,
        },
        "benchmark": {"symbol": bench_symbol, "name": bname},
        "weekly": weekly,
        "regime": regime,
        "liquidity": liq,
        "candles": candles,
        "zones": zone_view,
        "support_zone": sup,
        "fib": fib,
        "as_of": str(df.index[-1].date()),
    }
    if include_chart:
        result["chart"] = _chart_payload(df.tail(180), zone_view)
    return result


def _chart_payload(df: pd.DataFrame, zone_view: Optional[dict] = None) -> dict[str, Any]:
    dates = [str(i.date()) for i in df.index]

    def series(col):
        if col not in df.columns:
            return []
        return [None if pd.isna(x) else round(float(x), 2) for x in df[col].tolist()]

    zl = []
    for role in ("support", "resistance"):
        for z in (zone_view or {}).get(role, []):
            zl.append({"low": z["low"], "high": z["high"], "role": role, "strength": z["strength"]})

    return {
        "dates": dates,
        "open": series("Open"),
        "high": series("High"),
        "low": series("Low"),
        "close": series("Close"),
        "volume": [None if pd.isna(x) else int(x) for x in df["Volume"].tolist()],
        "sma20": series("SMA20"),
        "sma50": series("SMA50"),
        "sma200": series("SMA200"),
        "rsi": [None if pd.isna(x) else round(float(x), 1) for x in df["RSI"].tolist()],
        "macd_hist": [
            None if pd.isna(x) else round(float(x), 4) for x in df["MACD_hist"].tolist()
        ],
        "zones": zl,
    }


def empty_error(ticker: str, message: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "symbol": ticker.replace(".NS", ""),
        "error": message,
        "buy_score": None,
        "zone": "error",
    }
