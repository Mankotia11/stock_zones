"""Deterministic synthetic OHLCV generators for tests (no network needed)."""
import numpy as np
import pandas as pd


def make_ohlcv(closes, seed=0, vol_base=1_000_000, start="2023-01-02", wick=0.006, vol_fn=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1] * (1 + rng.normal(0, 0.002, n - 1))
    hi = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, wick, n)))
    lo = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, wick, n)))
    vol = vol_base * (1 + rng.normal(0, 0.15, n)).clip(0.4, None)
    if vol_fn is not None:
        vol = vol * vol_fn(np.arange(n))
    return pd.DataFrame({"Open": opens, "High": hi, "Low": lo, "Close": closes, "Volume": vol.astype(float)}, index=idx)


def trend_with_pullback(n_up=330, pull=14, drift=0.0012, vol=0.011, seed=1, start_price=100.0):
    """Steady uptrend, then a pullback of `pull` bars (~8%), ending near a prior swing zone."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n_up)
    r2 = rng.normal(-0.006, 0.008, pull)
    px = start_price * np.exp(np.cumsum(np.concatenate([r, r2])))
    return px


def downtrend(n=330, drift=-0.0020, vol=0.012, seed=2, start_price=200.0):
    rng = np.random.default_rng(seed)
    return start_price * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def parabolic(n=330, seed=3):
    rng = np.random.default_rng(seed)
    base = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n - 40)))
    spike = base[-1] * np.exp(np.cumsum(rng.normal(0.012, 0.008, 40)))
    return np.concatenate([base, spike])


def range_with_levels(n=400, seed=4, lo=90.0, hi=110.0):
    """Oscillates between planted support (lo) and resistance (hi)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    return mid + amp * np.sin(2 * np.pi * t / 50) * 0.92 + rng.normal(0, 0.4, n)


def piecewise(waypoints, noise=0.004, seed=0):
    """Linear path through (bar, price) waypoints with small multiplicative noise."""
    bars = np.arange(waypoints[-1][0] + 1)
    xs, ys = zip(*waypoints)
    path = np.interp(bars, xs, ys)
    rng = np.random.default_rng(seed)
    return path * (1 + rng.normal(0, noise, len(bars)))


# Staircase uptrend with swing lows planted at ~112, ~130, ~148 and highs at ~120/140/160/185.
UPTREND_WAYPOINTS = [
    (0, 100), (60, 120), (80, 112), (130, 140), (150, 130), (200, 160),
    (225, 148), (270, 185),
]


def uptrend_pullback_to_zone(end_price=150.0, seed=0, noise=0.004):
    """Uptrend, a sharp 15% drop, then a choppy base that finally dips into the ~148 swing-low zone."""
    tail = [(300, 158), (310, 164), (322, 153), (329, 158), (336, end_price)]
    return piecewise(UPTREND_WAYPOINTS + tail, noise=noise, seed=seed)


# A 160 high tested twice (bars 200 and 218), broken out ~bar 235, then price pulls back
# to retest it from above after peaking at 185.
RETEST_WAYPOINTS = [
    (0, 100), (60, 120), (80, 112), (130, 140), (150, 130), (200, 160),
    (210, 153), (218, 160), (225, 148), (270, 185), (298, 163), (304, 166),
]


def uptrend_retest(seed=0, noise=0.004):
    return piecewise(RETEST_WAYPOINTS, noise=noise, seed=seed)
