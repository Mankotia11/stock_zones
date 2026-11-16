import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import market
import store
import synth


@pytest.fixture(autouse=True)
def clean_cache():
    market.invalidate_cache()
    yield
    market.invalidate_cache()


def _frame(n=300):
    return synth.make_ohlcv(synth.uptrend_retest(seed=1), seed=1).iloc[:n]


def test_failed_download_is_not_cached_for_15_minutes(monkeypatch):
    calls = []

    def fake(*a, **k):
        calls.append(a[0])
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(market.yf, "download", fake)
    assert market.download_history("X.NS").empty
    assert market.download_history("X.NS").empty  # within FAIL_TTL: served from the short cache
    assert len(calls) == 1
    # age the failure entry past FAIL_TTL -> retried (it used to stay cached for the full 15 min)
    key = ("X.NS", "2y")
    ts, payload = market._price_cache[key]
    market._price_cache[key] = (ts - market.FAIL_TTL - 5, payload)
    market.download_history("X.NS")
    assert len(calls) == 2


def test_good_download_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(market.yf, "download", lambda *a, **k: calls.append(a[0]) or _frame())
    market.download_history("Y.NS")
    market.download_history("Y.NS")
    assert len(calls) == 1


def test_cache_is_keyed_by_period(monkeypatch):
    seen = []
    monkeypatch.setattr(market.yf, "download", lambda t, period=None, **k: seen.append(period) or _frame())
    market.download_history("Z.NS", period="2y")
    market.download_history("Z.NS", period="5y")
    assert seen == ["2y", "5y"]


@pytest.mark.parametrize("ticker,bench", [("RELIANCE.NS", "^NSEI"), ("AAPL", "^GSPC"), ("VOD.L", "^FTSE")])
def test_benchmark_follows_the_exchange(monkeypatch, ticker, bench):
    asked = []
    monkeypatch.setattr(market, "download_history", lambda t, period="2y": asked.append(t) or _frame())
    assert market.get_benchmark(ticker) is not None
    assert asked == [bench]


def test_analyze_ticker_end_to_end_offline(monkeypatch):
    monkeypatch.setattr(market.yf, "download", lambda *a, **k: _frame())
    monkeypatch.setattr(market, "fetch_info", lambda t: ({"shortName": "Test Co", "sector": "Tech"}, None))
    out = market.analyze_ticker("TESTCO")
    assert out.get("error") is None, out
    assert out["ticker"] == "TESTCO.NS" and out["name"] == "Test Co"
    assert out["benchmark"]["symbol"] == "^NSEI"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "w.db")
    import app as app_module

    return TestClient(app_module.app)


def test_api_watchlist_snapshot_and_backtest_validation(client, monkeypatch):
    monkeypatch.setattr(market.yf, "download", lambda *a, **k: _frame())
    monkeypatch.setattr(market, "fetch_info", lambda t: ({}, None))
    assert client.get("/api/watchlist").json() == {"tickers": []}
    r = client.post("/api/watchlist", json={"ticker": "abc"})
    assert r.status_code == 200 and r.json()["snapshot"]["ticker"] == "ABC.NS"
    snap = client.get("/api/snapshot").json()
    assert snap["count"] == 1 and snap["stocks"][0]["zones"] is not None
    assert client.get("/api/backtest/ABC?period=1d").status_code == 400
    assert client.get("/api/backtest/ABC?step=0").status_code == 400
    assert client.delete("/api/watchlist/ABC.NS").json() == {"ok": True}
