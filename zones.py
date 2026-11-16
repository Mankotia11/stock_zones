#!/usr/bin/env python3
"""Support/resistance zone engine, ATR-scaled swing detection and candle checks.

Pure functions over OHLCV frames (no network, no globals), so they can be
unit-tested and replayed bar-by-bar in the backtester without lookahead.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

PIVOT_ORDER = 3  # bars either side to confirm a fractal swing point
ZONE_LOOKBACK = 320  # bars of history used to build zones
RECENCY_HALF_LIFE = 120  # a touch loses half its weight every 120 bars
MIN_TOUCH_GAP = 5  # pivots closer than this are the same "touch event"
REACTION_ATR = 1.5  # a touch must be followed by a reversal this many ATRs...
REACTION_BARS = 10  # ...within this many bars
MIN_ZONE_AGE = 15  # a zone must pre-date a break/retest by this many bars
ZIGZAG_ATR_MULT = 2.5  # a swing must retrace this many ATRs to be confirmed
ZIGZAG_MIN_PCT = 0.03  # ...and at least this fraction of price


# --------------------------------------------------------------------------
# Fractal pivots
# --------------------------------------------------------------------------
def _dedupe(idx: np.ndarray, gap: int) -> np.ndarray:
    keep: list[int] = []
    for i in idx:
        if keep and int(i) - keep[-1] <= gap:
            continue  # plateau tie: same extreme value, keep the first bar
        keep.append(int(i))
    return np.array(keep, dtype=int)


def pivot_idx(series: pd.Series, order: int = PIVOT_ORDER, mode: str = "low") -> np.ndarray:
    """Positional indices of confirmed fractal swing lows/highs.

    The centred window is NaN for the last `order` bars, so unconfirmed pivots
    at the right edge are excluded. Equal-value plateaus yield one pivot.
    """
    win = 2 * order + 1
    roll = series.rolling(win, center=True)
    ext = roll.min() if mode == "low" else roll.max()
    mask = ((series == ext) & ext.notna()).to_numpy()
    return _dedupe(np.flatnonzero(mask), order)


def atr_last(df: pd.DataFrame) -> float:
    if "ATR" in df.columns:
        v = df["ATR"].iloc[-1]
        if np.isfinite(v) and v > 0:
            return float(v)
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = tr.tail(14).mean()
    if np.isfinite(v) and v > 0:
        return float(v)
    return float(c.iloc[-1]) * 0.02


# --------------------------------------------------------------------------
# ATR zig-zag (significant swings only)
# --------------------------------------------------------------------------
def zigzag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    mult: float = ZIGZAG_ATR_MULT,
    min_pct: float = ZIGZAG_MIN_PCT,
) -> list[dict[str, Any]]:
    """Confirmed swing highs/lows. A pivot is confirmed only after price has
    reversed by max(mult*ATR, min_pct*price) from the extreme, so the newest
    pivot is never based on future bars."""
    n = len(high)
    piv: list[dict[str, Any]] = []
    if n < 5:
        return piv

    def thr(i: int) -> float:
        a = atr[i] if np.isfinite(atr[i]) else 0.0
        return max(mult * a, min_pct * close[i])

    direction = 0  # 1 = up-leg, -1 = down-leg, 0 = undecided
    hi_p, hi_i = high[0], 0
    lo_p, lo_i = low[0], 0
    for i in range(1, n):
        t = thr(i)
        if direction == 0:
            if high[i] > hi_p:
                hi_p, hi_i = high[i], i
            if low[i] < lo_p:
                lo_p, lo_i = low[i], i
            if hi_p - low[i] >= t and hi_i < i and hi_i != lo_i:
                piv.append({"i": hi_i, "p": float(hi_p), "t": "H"})
                direction, lo_p, lo_i = -1, low[i], i
            elif high[i] - lo_p >= t and lo_i < i and hi_i != lo_i:
                piv.append({"i": lo_i, "p": float(lo_p), "t": "L"})
                direction, hi_p, hi_i = 1, high[i], i
        elif direction == 1:
            if high[i] > hi_p:
                hi_p, hi_i = high[i], i
            elif hi_p - low[i] >= t:
                piv.append({"i": hi_i, "p": float(hi_p), "t": "H"})
                direction, lo_p, lo_i = -1, low[i], i
        else:
            if low[i] < lo_p:
                lo_p, lo_i = low[i], i
            elif high[i] - lo_p >= t:
                piv.append({"i": lo_i, "p": float(lo_p), "t": "L"})
                direction, hi_p, hi_i = 1, high[i], i
    return piv


# --------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------
def find_zones(df: pd.DataFrame, lookback: int = ZONE_LOOKBACK, order: int = PIVOT_ORDER) -> list[dict[str, Any]]:
    """Cluster swing highs AND lows into price zones (a level that was
    resistance can become support, so both feed the same clusters).

    Zone width and clustering tolerance scale with ATR. Strength rewards
    repeated, recent, volume-backed touches and role reversal; it penalises
    levels that price has sliced through many times (chop).
    """
    n = len(df)
    if n < 40:
        return []
    price = float(df["Close"].iloc[-1])
    atr = atr_last(df)
    start = max(0, n - lookback)
    high, low = df["High"], df["Low"]
    closes = df["Close"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    vol_base = pd.Series(vol).rolling(50, min_periods=10).median().to_numpy()

    hi_arr = high.to_numpy(dtype=float)
    lo_arr = low.to_numpy(dtype=float)
    atr_arr = df["ATR"].to_numpy(dtype=float) if "ATR" in df.columns else np.full(n, atr)

    seeds: list[tuple[float, int, str, float]] = []
    for kind, ser, mode in (("H", high, "high"), ("L", low, "low")):
        for i in pivot_idx(ser, order, mode):
            if i < start:
                continue
            # Reaction filter: a "touch" only counts if price actually reversed
            # >= REACTION_ATR ATRs within the next REACTION_BARS bars. This drops
            # the meaningless fractals that litter a steep one-way move.
            a_i = atr_arr[i] if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else atr
            fwd = slice(i + 1, min(n, i + 1 + REACTION_BARS))
            if fwd.start >= n:
                continue
            if kind == "L":
                reaction = hi_arr[fwd].max() - lo_arr[i]
            else:
                reaction = hi_arr[i] - lo_arr[fwd].min()
            if reaction < REACTION_ATR * a_i:
                continue
            vb = vol_base[i]
            if np.isfinite(vb) and vb > 0:
                vr = float(vol[max(0, i - 1) : i + 2].mean() / vb)
            else:
                vr = 1.0
            seeds.append((float(ser.iloc[i]), int(i), kind, float(min(max(vr, 0.3), 3.0))))
    if not seeds:
        return []

    seeds.sort(key=lambda s: s[0])
    tol = float(np.clip(0.55 * atr, 0.004 * price, 0.03 * price))
    clusters: list[list[tuple[float, int, str, float]]] = []
    cur = [seeds[0]]
    for s in seeds[1:]:
        if s[0] - cur[-1][0] <= tol and s[0] - cur[0][0] <= 2.2 * tol:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    clusters.append(cur)

    zones: list[dict[str, Any]] = []
    for cl in clusters:
        prices = np.array([s[0] for s in cl])
        idxs = np.sort(np.array([s[1] for s in cl]))
        lo, hi = float(prices.min()), float(prices.max())
        min_w = 0.3 * atr
        if hi - lo < min_w:
            c = (hi + lo) / 2
            lo, hi = c - min_w / 2, c + min_w / 2
        else:
            lo, hi = lo - 0.08 * atr, hi + 0.08 * atr

        touches = 1 + int((np.diff(idxs) >= MIN_TOUCH_GAP).sum())
        first, last_i = int(idxs[0]), int(idxs[-1])
        w = np.array([0.5 ** ((n - 1 - s[1]) / RECENCY_HALF_LIFE) for s in cl])
        mid = float(np.average(prices, weights=w + 1e-9))
        mid = min(max(mid, lo), hi)

        # How often did closes pass fully through the zone after it formed?
        seg = closes[first:]
        state = np.zeros(len(seg), dtype=int)
        state[seg > hi + 0.25 * atr] = 1
        state[seg < lo - 0.25 * atr] = -1
        nz = state[state != 0]
        flips = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0

        vol_avg = float(np.mean([s[3] for s in cl]))
        recency = float(0.5 ** ((n - 1 - last_i) / RECENCY_HALF_LIFE))
        vol_bonus = float(np.clip(10 * (min(vol_avg, 2.0) - 1.0), -4, 10))
        flip_bonus = 10.0 if 1 <= flips <= 3 else (-12.0 if flips >= 4 else 0.0)
        strength = float(np.clip(18 * min(touches, 5) + 20 * recency + vol_bonus + flip_bonus, 0, 100))

        role = "support" if price > hi else ("resistance" if price < lo else "inside")
        zones.append(
            {
                "low": round(lo, 2),
                "high": round(hi, 2),
                "mid": round(mid, 2),
                "touches": touches,
                "strength": int(round(strength)),
                "flips": flips,
                "age": n - 1 - first,
                "last_touch_age": n - 1 - last_i,
                "role": role,
            }
        )
    zones.sort(key=lambda z: z["mid"])
    return zones


def pick_support(zones: list[dict], price: float, atr: float, max_atr: float = 12.0) -> Optional[dict]:
    """Best demand zone at/below price: strong beats weak, near beats far
    (9 strength points per ATR of distance)."""
    best, best_sc = None, -1e9
    for z in zones:
        if z["low"] >= price:
            continue
        gap = max(0.0, price - z["high"]) / atr
        if gap > max_atr:
            continue
        sc = z["strength"] - 9.0 * gap
        if sc > best_sc:
            best, best_sc = z, sc
    return best


def pick_resistances(zones: list[dict], price: float, atr: float) -> list[dict]:
    """Supply zones overhead, nearest first. Weak one-touch zones are skipped
    unless nothing stronger exists."""
    above = sorted((z for z in zones if z["mid"] > price + 0.15 * atr), key=lambda z: z["mid"])
    strong = [z for z in above if z["strength"] >= 30]
    return strong or above


def recent_breakdown(df: pd.DataFrame, zones: list[dict], atr: float, bars: int = 12) -> Optional[dict]:
    """A meaningful zone that closes were above within `bars` and are now
    decisively below: support has failed and should not be bought."""
    if len(df) < bars + 2:
        return None
    closes = df["Close"].to_numpy(dtype=float)
    best = None
    for z in zones:
        # Must be a level that was tested before (not one the fall itself created).
        if z["strength"] < 35 or z["touches"] < 2 or z["age"] < MIN_ZONE_AGE + bars:
            continue
        if closes[-1] < z["low"] - 0.25 * atr and (closes[-bars - 1 : -1] > z["high"]).any():
            if best is None or z["strength"] > best["strength"]:
                best = z
    return best


def breakout_retest(df: pd.DataFrame, zones: list[dict], atr: float, max_age: int = 30) -> Optional[dict]:
    """Price broke above a zone on real volume and is now pulling back to test
    it from above (old resistance turned support)."""
    n = len(df)
    if n < max_age + 5:
        return None
    closes = df["Close"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    vbase = pd.Series(vol).rolling(20).mean().shift(1).to_numpy()
    price = closes[-1]
    best = None
    for z in zones:
        if z["strength"] < 35 or price <= z["high"] - 0.25 * atr:
            continue
        above = closes > z["high"] + 0.25 * atr
        crosses = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        crosses = crosses[crosses >= n - max_age]
        if not len(crosses):
            continue
        j = int(crosses[-1])
        if (n - 1 - z["age"]) > j - MIN_ZONE_AGE:
            continue  # the zone must have existed before the breakout
        if (closes[j:] < z["low"] - 0.25 * atr).any():
            continue
        if highs[j:].max() < z["high"] + 0.8 * atr:
            continue  # never really left the zone
        if not (z["high"] - 0.25 * atr <= price <= z["high"] + 1.2 * atr):
            continue
        seg = slice(j, min(j + 2, n))
        base = vbase[seg]
        ok = np.isfinite(base) & (base > 0)
        vol_confirm = bool(ok.any() and (vol[seg][ok] / base[ok]).max() >= 1.25)
        cand = {"zone": z, "bars_ago": n - 1 - j, "vol_confirm": vol_confirm}
        if best is None or z["strength"] > best["zone"]["strength"]:
            best = cand
    return best


# --------------------------------------------------------------------------
# Candle confirmation
# --------------------------------------------------------------------------
def candle_signals(df: pd.DataFrame, zone: Optional[dict], atr: float) -> dict[str, bool]:
    out = {"rejection": False, "engulfing": False, "reclaim": False, "close_below_zone": False}
    if zone is None or len(df) < 4:
        return out
    o = df["Open"].to_numpy(dtype=float)[-3:]
    h = df["High"].to_numpy(dtype=float)[-3:]
    l = df["Low"].to_numpy(dtype=float)[-3:]
    c = df["Close"].to_numpy(dtype=float)[-3:]

    def hammer(i: int) -> bool:
        rng = h[i] - l[i]
        if rng <= 0:
            return False
        body = abs(c[i] - o[i])
        lower = min(o[i], c[i]) - l[i]
        return bool(lower >= 0.5 * rng and lower >= 1.5 * body and (c[i] - l[i]) >= 0.6 * rng)

    touched = min(l[-2:]) <= zone["high"] + 0.25 * atr
    out["rejection"] = bool(touched and (hammer(-1) or (hammer(-2) and c[-1] >= c[-2])))
    out["engulfing"] = bool(
        touched and c[-2] < o[-2] and c[-1] > o[-1] and c[-1] >= o[-2] and o[-1] <= c[-2]
    )
    out["reclaim"] = bool(min(l) < zone["low"] - 0.1 * atr and c[-1] > zone["low"] and c[-1] >= c[-2])
    out["close_below_zone"] = bool(c[-1] < zone["low"] - 0.5 * atr)
    return out
