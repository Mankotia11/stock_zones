import numpy as np
import pandas as pd

import synth
from backtest import _outcome, dedupe_signals, summarize, walk_forward


def arrs(rows):
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return o, h, l, c


def test_outcome_stop_target_gap_and_tie():
    # bars: signal bar t=0, then next bars (o, h, l, c)
    bars = [(100, 100, 100, 100), (100, 103, 99, 102), (102, 111, 101, 110)]
    o, h, l, c = arrs(bars)
    assert _outcome(o, h, l, c, 0, 100, 95, 110, 2)[0] == "target"
    # stop first
    bars = [(100, 100, 100, 100), (100, 101, 94, 96), (96, 120, 95, 118)]
    o, h, l, c = arrs(bars)
    kind, r = _outcome(o, h, l, c, 0, 100, 95, 110, 2)
    assert kind == "stop" and round(r, 2) == -1.0
    # same bar touches both: conservative = stop
    bars = [(100, 100, 100, 100), (100, 112, 94, 100), (100, 100, 100, 100)]
    o, h, l, c = arrs(bars)
    assert _outcome(o, h, l, c, 0, 100, 95, 110, 2)[0] == "stop"
    # gap down through the stop fills at the open (worse than -1R)
    bars = [(100, 100, 100, 100), (90, 91, 88, 89), (89, 90, 88, 89)]
    o, h, l, c = arrs(bars)
    kind, r = _outcome(o, h, l, c, 0, 100, 95, 110, 2)
    assert kind == "stop" and r < -1.5
    # timeout and insufficient data
    bars = [(100, 100, 100, 100), (100, 102, 99, 101), (101, 103, 100, 102)]
    o, h, l, c = arrs(bars)
    assert _outcome(o, h, l, c, 0, 100, 95, 110, 2)[0] == "timeout"
    assert _outcome(o, h, l, c, 0, 100, 95, 110, 5)[0] is None
    # no stop defined / entry already below stop -> no outcome
    assert _outcome(o, h, l, c, 0, 100, None, 110, 2) == (None, None)
    assert _outcome(o, h, l, c, 0, 94, 95, 110, 2) == (None, None)


def _series(n=700, seed=11):
    rng = np.random.default_rng(seed)
    return synth.make_ohlcv(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.013, n))), seed=seed)


def test_walk_forward_shape_and_columns():
    df = _series()
    res = walk_forward("T.NS", df, step=20)
    assert len(res) >= 15
    assert {"date", "score", "zone", "setup", "entry", "ret_10", "ret_20", "ret_40", "mae_20", "outcome", "r"} <= set(res.columns)
    assert (res["mae_20"] <= 0).mean() > 0.9  # worst excursion from the entry is a drawdown
    # entries are the NEXT bar's open, never the signal bar's close
    row = res.iloc[3]
    pos = df.index.get_loc(row["date"])
    assert row["entry"] == df["Open"].iloc[pos + 1]


def test_walk_forward_has_no_lookahead():
    """Scrambling every bar after cut T must not change anything the model decided before T."""
    df = _series()
    cut = 520
    rng = np.random.default_rng(99)
    df2 = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        df2.iloc[cut:, df2.columns.get_loc(col)] = df2[col].iloc[cut:].to_numpy() * rng.uniform(0.7, 1.3, len(df2) - cut)
    a = walk_forward("T.NS", df, step=10)
    b = walk_forward("T.NS", df2, step=10)
    before = a["date"] < df.index[cut - 1]
    assert before.sum() >= 10
    pd.testing.assert_frame_equal(
        a.loc[before, ["date", "score", "zone", "setup", "invalidation", "target_1", "rr"]].reset_index(drop=True),
        b.loc[before, ["date", "score", "zone", "setup", "invalidation", "target_1", "rr"]].reset_index(drop=True),
    )


def test_summarize_and_dedupe():
    df = _series()
    res = walk_forward("T.NS", df, step=10)
    s = summarize(res, cooldown_bars=10, step=10)
    assert s["samples"] == len(res)
    assert set(s) >= {"baseline_all_dates", "by_zone", "by_setup", "by_score_bucket", "buy_signals_deduped"}
    assert s["baseline_all_dates"]["edge_vs_all_20d"] == 0
    # dedupe: 15 signals sampled every 3 bars with a 10-bar cooldown -> every 4th is kept (0,4,8,12)
    dates = pd.bdate_range("2024-01-01", periods=15)
    sig = pd.DataFrame({"ticker": ["A"] * 15, "date": dates})
    assert list(dedupe_signals(sig, cooldown_bars=10, step=3).index) == [0, 4, 8, 12]
    # cooldown is tracked per ticker
    two = pd.concat([sig, sig.assign(ticker="B")], ignore_index=True)
    assert len(dedupe_signals(two, cooldown_bars=10, step=3)) == 8
