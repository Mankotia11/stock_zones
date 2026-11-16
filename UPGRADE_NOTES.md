# stock_zones v2: what changed and why

## Logic gaps found, and the fix for each

| Gap in v1 | v2 |
|---|---|
| "Support" was one swing-low price; resistance was the last swing high (which could sit *below* price and silently null the R:R) | `zones.py`: swing highs **and** lows clustered into ATR-wide zones with touches, recency, volume, role-reversal. Support and resistance are both zones. |
| Buy band / stop were fixed percentages (`support*0.992`, `*0.97`) regardless of volatility | Band = zone edges; stop = zone low minus 0.6 ATR (min 1%); "near a MA" scales with ATR% |
| No reward side: R:R used the last swing high | Target 1/2 = next *significant* supply zones; blue-sky falls back to a projected 2.5 ATR target and says so |
| Score could be "buy grade" with no defined entry/stop or poor R:R | Gates: no zone -> capped at Watch; R:R < 1 -> Watch; strong buy needs R:R >= 1.5 |
| 3-bar fractal pivots drove structure, divergence and Fib (noise on any wiggle; ties produced duplicate pivots) | ATR zig-zag swings (>= 2.5 ATR and >= 3%); live break of the last swing counts so structure doesn't lag; plateau ties deduped |
| RSI divergence accepted any two lows 7 bars apart | Needs 5-90 bars gap, lower low >= 0.25 ATR, RSI +2 pts, low still intact. Bearish divergence now also detected |
| Zone seeds from raw fractals created fake "support" inside steep one-way moves | Reaction filter: a touch only counts if price reversed >= 1.5 ATR within 10 bars |
| Benchmark was always Nifty, even for AAPL | Per-exchange benchmark (^NSEI, ^GSPC, ^FTSE, ...) and index regime (bull/neutral/bear) |
| Partial-bar drop used IST 15:35 for every market | Per-exchange session close + timezone |
| Failed/empty download cached for 15 min; cache ignored `period` | Failures retry after 60 s; cache keyed by (ticker, period) |
| No higher-timeframe view | Weekly trend (30-week MA + slope, weekly RSI) feeds trend score and the Quality-pullback gate |
| No liquidity check (circuit-locked / thin smallcaps) | ADV traded value + zero-range-day check -> flag, score penalty |
| No price-action confirmation | Rejection wick / bullish engulfing / reclaim-after-sweep at the zone |
| Missing setups | **Breakout retest** (old resistance as support); **support failed** -> Breakdown; Reversal now requires a level to reverse from |
| Stale data silently scored | > 10 calendar days old -> Neutral with a warning |
| Weights were guesses with no way to check them | `backtest.py` (below) |

## New output fields (all additive; old keys unchanged)
`zones{support,resistance}`, `support_zone`, `targets{entry_ref,target_1,target_2,source}`, `risk_pct`,
`size_pct_for_1pct_risk`, `weekly`, `regime`, `liquidity`, `candles`, `benchmark`, `stale_days`,
`scores.risk` (5th sub-score; weights now 28/24/20/12/16), `technicals.{atr, rs_60, rs_line_up, rsi_bear_divergence}`.
`chart.zones` feeds zone bands, buy band, stop and target lines on the chart.

## Validate the score on real data (needs network)
```
python backtest.py RELIANCE TCS HDFCBANK INFY --period 7y --step 3
GET /api/backtest/RELIANCE?period=5y&step=5
```
Each sampled date is analysed with data up to that date only, entry = next bar's open. Reports forward returns,
worst drawdown, stop/target first-touch and mean R by zone, setup and score bucket, vs an all-dates baseline.
Look at `edge_vs_all_20d` and `mean_R`, and ignore rows marked `low n`. A hand-picked watchlist is survivorship-biased;
for a fair read use a broad index constituent list.

**Null test (synthetic random walks, no real edge exists; 36 series, ~5,300 samples):** the 20-day edge of buy
signals vs the all-dates baseline stayed between -0.33 and +0.08 pts, and mean R was ~0 at zero drift. So the harness
does not invent edge. `tests/test_backtest.py` also proves decisions are causal (scrambling every bar after date T
changes nothing decided before T). None of this says the score works on real markets; run it on real data for that.

## Tuning knobs
`zones.py`: `REACTION_ATR`, `ZIGZAG_ATR_MULT`, `RECENCY_HALF_LIFE`. `analysis.py`: score weights, `STALE_DAYS`.
`context.py`: liquidity thresholds per region. Change one at a time and re-run the backtest.

## Tests
`pip install -r requirements-dev.txt && pytest` (39 tests, offline, ~8 s).

## Housekeeping
`.gitignore` added. `watchlist.db` and `__pycache__/` were committed in v1; to stop tracking them without deleting
your local copies: `git rm -r --cached __pycache__ watchlist.db`.
