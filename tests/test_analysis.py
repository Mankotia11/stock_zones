import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import synth
from analysis import BUY_ZONES, analyze_frame, clean_ohlcv, swing_points
from context import benchmark_for, region_of, session_is_open_for_bar

LEGACY_KEYS = {
    "ticker", "symbol", "name", "sector", "price", "change_1d", "change_20d", "buy_score", "zone",
    "setup", "signals", "buy_from", "buy_to", "invalidation", "support", "resistance", "risk_reward",
    "in_buy_band", "limited_history", "scores", "technicals", "fundamentals", "fib", "as_of", "chart",
}


def run(px, ticker="TEST.NS", bench=None, seed=0, **kw):
    df = synth.make_ohlcv(px, seed=seed, **kw)
    if bench is None:
        bench = synth.make_ohlcv(synth.piecewise([(0, 100), (len(px) - 1, 150)], noise=0.004, seed=5), seed=5)
    return analyze_frame(ticker, df, bench=bench, drop_partial=False, now=df.index[-1].date())


def test_legacy_output_contract_preserved():
    a = run(synth.uptrend_retest(seed=1), seed=1)
    assert LEGACY_KEYS <= set(a)
    assert {"location", "momentum", "trend", "volume"} <= set(a["scores"])
    json.dumps(a, allow_nan=False)  # FastAPI-safe: no numpy scalars, no NaN


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_healthy_pullback_into_retested_zone_is_a_buy_zone(seed):
    a = run(synth.uptrend_retest(seed=seed), seed=seed)
    assert a["zone"] in BUY_ZONES
    assert a["setup"] in ("Quality pullback", "Breakout retest")
    assert a["invalidation"] < a["buy_from"] < a["buy_to"]
    assert a["targets"]["target_1"] > a["targets"]["entry_ref"] > a["invalidation"]
    assert a["risk_reward"] > 0 and a["size_pct_for_1pct_risk"] > 0
    assert a["weekly"]["trend"] == "up"


def test_straight_line_crash_is_avoid_not_buy():
    a = run(synth.uptrend_pullback_to_zone(end_price=140.0), seed=1)
    assert a["zone"] == "avoid"
    assert a["buy_from"] is None and a["invalidation"] is None and a["risk_reward"] is None


def test_parabolic_spike_is_extended():
    a = run(synth.parabolic(), seed=3)
    assert a["zone"] == "extended"


def test_stale_data_gives_no_signal():
    px = synth.uptrend_retest(seed=1)
    df = synth.make_ohlcv(px, seed=1)
    a = analyze_frame("TEST.NS", df, drop_partial=False, now=df.index[-1].date() + timedelta(days=30))
    assert a["zone"] == "neutral" and a["stale_days"] == 30
    assert any("stale" in s.lower() for s in a["signals"])


def test_buy_zone_always_has_defined_entry_stop_and_acceptable_rr():
    rng = np.random.default_rng(7)
    seen_buy = 0
    for k in range(60):
        n = int(rng.integers(120, 480))
        px = 100 * np.exp(np.cumsum(rng.normal(rng.normal(0, 0.002), rng.uniform(0.004, 0.04), n)))
        a = run(px, seed=k)
        json.dumps(a, allow_nan=False)
        if a["zone"] in BUY_ZONES:
            seen_buy += 1
            assert a["buy_from"] is not None and a["invalidation"] < a["buy_from"] < a["buy_to"]
            assert a["risk_reward"] is None or a["risk_reward"] >= 1.0
            if a["zone"] == "strong_buy_zone":
                assert a["risk_reward"] is not None and a["risk_reward"] >= 1.5
    assert seen_buy >= 1  # the fuzz set actually exercised the buy path


def test_bands_and_stops_scale_with_volatility():
    px = synth.uptrend_retest(seed=1)
    calm = run(px, seed=1, wick=0.004)
    wild = run(px, seed=1, wick=0.014)
    assert wild["technicals"]["atr"] > calm["technicals"]["atr"] * 1.5
    if calm["invalidation"] and wild["invalidation"]:
        d = lambda a: a["buy_from"] - a["invalidation"]
        assert d(wild) > d(calm)


def test_bearish_index_regime_lowers_score():
    px = synth.uptrend_retest(seed=1)
    bull = run(px, seed=1)
    bear_bench = synth.make_ohlcv(synth.piecewise([(0, 150), (len(px) - 1, 90)], noise=0.004, seed=6), seed=6)
    bear = run(px, seed=1, bench=bear_bench)
    assert bull["regime"]["state"] == "bull" and bear["regime"]["state"] == "bear"
    assert bear["buy_score"] < bull["buy_score"]


def test_thin_liquidity_is_flagged_and_penalised():
    px = synth.uptrend_retest(seed=1)
    liquid = run(px, seed=1, vol_base=1_000_000)
    thin = run(px, seed=1, vol_base=100_000)  # ~Rs 1.6 Cr/day
    assert liquid["liquidity"]["state"] == "ok" and thin["liquidity"]["state"] == "thin"
    assert thin["buy_score"] <= liquid["buy_score"]


def test_us_ticker_is_scored_against_sp500_not_nifty():
    a = run(synth.uptrend_retest(seed=1), ticker="AAPL", seed=1)
    assert a["benchmark"]["symbol"] == "^GSPC" and a["benchmark"]["name"] == "S&P 500"
    assert not any("Nifty" in s for s in a["signals"])


def test_benchmark_and_region_mapping():
    assert benchmark_for("RELIANCE.NS") == "^NSEI"
    assert benchmark_for("TCS.BO") == "^NSEI"
    assert benchmark_for("AAPL") == "^GSPC"
    assert benchmark_for("BRK.B") == "^GSPC"  # class-share dot is not an exchange suffix
    assert benchmark_for("VOD.L") == "^FTSE"
    assert region_of("^NSEI") == "IN" and region_of("^GSPC") == "US"


def test_partial_bar_rule_is_exchange_aware():
    ist = datetime.fromisoformat("2026-11-16T10:00:00+05:30")  # NSE open, US still closed (prev. session)
    today = pd.Timestamp("2026-11-16")
    assert session_is_open_for_bar(today, "IN", now=ist)  # NSE bar for today still forming
    assert not session_is_open_for_bar(today, "IN", now=datetime.fromisoformat("2026-11-16T16:00:00+05:30"))
    # 21:00 IST is 10:30 ET: a bar dated today on a US ticker is still forming (old IST-only rule missed this).
    assert session_is_open_for_bar(today, "US", now=datetime.fromisoformat("2026-11-16T21:00:00+05:30"))
    # a bar from a previous date is never partial
    assert not session_is_open_for_bar(pd.Timestamp("2026-11-13"), "IN", now=ist)


def test_clean_ohlcv_repairs_bad_prints_and_dedupes():
    df = synth.make_ohlcv(np.linspace(100, 120, 100))
    df.iloc[10, df.columns.get_loc("High")] = df.iloc[10]["Low"] - 5  # high below low
    df.iloc[20, df.columns.get_loc("Open")] = np.nan
    df = pd.concat([df, df.iloc[[-1]]])  # duplicated last bar
    out = clean_ohlcv(df, ticker="X.NS", drop_partial=False)
    assert len(out) == 100
    assert (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)).all()
    assert out["Open"].notna().all()


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        run(np.linspace(100, 110, 50))


def test_swing_points_backcompat():
    s = pd.Series(np.sin(np.linspace(0, 12, 120)))
    assert len(swing_points(s)) >= 4
