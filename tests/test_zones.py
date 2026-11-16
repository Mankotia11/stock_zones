import numpy as np
import pandas as pd

import synth
from analysis import add_indicators
from zones import (
    atr_last,
    breakout_retest,
    candle_signals,
    find_zones,
    pivot_idx,
    recent_breakdown,
    zigzag,
)


def _prep(px, **kw):
    return add_indicators(synth.make_ohlcv(px, **kw))


def test_pivot_plateau_yields_single_pivot():
    s = pd.Series([5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    idx = pivot_idx(s, order=3, mode="low")
    assert list(idx) == [4]  # the two equal lows collapse to the first bar


def test_planted_support_and_resistance_found_with_multiple_touches():
    df = _prep(synth.range_with_levels(n=400))
    zones = find_zones(df)
    sup = [z for z in zones if abs(z["mid"] - 91) <= 2]
    res = [z for z in zones if abs(z["mid"] - 109) <= 2]
    assert sup and res
    assert max(z["touches"] for z in sup) >= 3
    assert max(z["touches"] for z in res) >= 3
    assert max(z["strength"] for z in sup) >= 60


def test_zone_width_scales_with_atr():
    calm = _prep(synth.range_with_levels(n=400), wick=0.002)
    wild = _prep(synth.range_with_levels(n=400), wick=0.02)
    w = lambda d: np.mean([z["high"] - z["low"] for z in find_zones(d)])
    assert w(wild) > w(calm)


def test_steep_one_way_move_does_not_create_noise_zones():
    # Straight decline with tiny noise: the reaction filter should leave (almost) no zones.
    df = _prep(synth.piecewise([(0, 200), (200, 100)], noise=0.002, seed=3))
    strong = [z for z in find_zones(df) if z["touches"] >= 2]
    assert len(strong) <= 1


def test_zigzag_is_causal():
    """Pivots found on a prefix must be exactly the first pivots found on the full series."""
    df = _prep(synth.uptrend_retest(seed=2), seed=2)
    args = lambda d: [d[c].to_numpy(float) for c in ("High", "Low", "Close", "ATR")]
    full = zigzag(*args(df))
    for cut in (150, 220, 280):
        part = zigzag(*args(df.iloc[:cut]))
        assert part == full[: len(part)]


def test_indicators_have_no_lookahead():
    df = synth.make_ohlcv(synth.uptrend_retest(seed=1), seed=1)
    full = add_indicators(df)
    for cut in (120, 200, 250):
        part = add_indicators(df.iloc[:cut])
        for col in ("SMA20", "SMA200", "RSI", "ATR", "ADX", "MACD_hist", "CMF", "Rel_Vol", "BB_pct", "STOCH_K"):
            a, b = part[col].iloc[-1], full[col].iloc[cut - 1]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (col, cut)


def test_breakout_retest_detected_on_planted_pattern():
    df = _prep(synth.uptrend_retest(seed=1), seed=1)
    r = breakout_retest(df, find_zones(df), atr_last(df))
    assert r is not None
    assert 158 <= r["zone"]["low"] <= 163 and r["bars_ago"] <= 30


def test_recent_breakdown_detected_when_tested_support_fails():
    px = np.concatenate([synth.range_with_levels(n=300), np.linspace(91, 80, 14)])
    df = _prep(px)
    z = recent_breakdown(df, find_zones(df), atr_last(df))
    assert z is not None and z["touches"] >= 3


def test_no_breakdown_inside_a_healthy_range():
    df = _prep(synth.range_with_levels(n=300))
    assert recent_breakdown(df, find_zones(df), atr_last(df)) is None


def test_candle_rejection_and_reclaim():
    zone = {"low": 99.0, "high": 101.0}
    base = dict(
        Open=[103, 102, 101.5], High=[104, 103, 102.0], Low=[102, 100.5, 96.5], Close=[102.5, 101.2, 101.4],
        Volume=[1e6] * 3,
    )
    df = pd.DataFrame(base, index=pd.bdate_range("2024-01-01", periods=3))
    df = pd.concat([df] * 2).iloc[-4:]  # >= 4 rows
    df.index = pd.bdate_range("2024-01-01", periods=len(df))
    c = candle_signals(df, zone, atr=1.5)
    assert c["rejection"] and c["reclaim"] and not c["close_below_zone"]
    assert not any(candle_signals(df, None, 1.5).values())
