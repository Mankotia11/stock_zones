const ZONE_LABEL = {
  strong_buy_zone: "Strong buy zone",
  buy_zone: "Buy zone",
  watch: "Watch",
  neutral: "Neutral",
  extended: "Extended",
  avoid: "Avoid",
  error: "Error",
};

let SNAPSHOT = { stocks: [], errors: [] };
let SELECTED = null;

const $ = (id) => document.getElementById(id);

function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

function mcap(n) {
  if (!n) return "—";
  if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (n >= 1e7) return (n / 1e7).toFixed(1) + " Cr";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  return String(n);
}

function clsChg(n) {
  if (n > 0) return "up";
  if (n < 0) return "down";
  return "";
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}

function setStatus(msg) {
  $("status").textContent = msg || "";
}

function renderKpis(data) {
  const stocks = data.stocks || [];
  const inZone = stocks.filter((s) => s.zone === "strong_buy_zone" || s.zone === "buy_zone").length;
  const watch = stocks.filter((s) => s.zone === "watch").length;
  const avoid = stocks.filter((s) => s.zone === "avoid" || s.zone === "extended").length;
  $("kpis").innerHTML = [
    kpi("Watchlist", stocks.length),
    kpi("In buy zone", inZone),
    kpi("Watch", watch),
    kpi("Extended / avoid", avoid),
  ].join("");
}

function kpi(lbl, val) {
  return `<div class="kpi"><div class="lbl">${lbl}</div><div class="val">${val}</div></div>`;
}

function renderTable(data) {
  const body = $("rows");
  const stocks = data.stocks || [];
  if (!stocks.length) {
    body.innerHTML = `<tr><td colspan="9" class="empty">Add a ticker to start monitoring buy zones. Indian symbols resolve on NSE (RELIANCE). Use AAPL or MSFT for US listings.</td></tr>`;
    return;
  }
  body.innerHTML = stocks
    .map((s) => {
      const band =
        s.buy_from && s.buy_to
          ? `${fmt(s.buy_from)} – ${fmt(s.buy_to)}${s.in_buy_band ? "  in" : ""}`
          : "—";
      const active = SELECTED === s.ticker ? "active" : "";
      return `<tr class="${active}" data-ticker="${s.ticker}">
        <td class="sym">${s.symbol}<small>${s.sector || ""}</small></td>
        <td><span class="pill ${s.zone}">${ZONE_LABEL[s.zone] || s.zone}</span></td>
        <td class="num">${fmt(s.buy_score, 1)}</td>
        <td class="num">${fmt(s.price)}</td>
        <td class="num ${clsChg(s.change_1d)}">${s.change_1d > 0 ? "+" : ""}${fmt(s.change_1d)}</td>
        <td class="num">${fmt(s.technicals?.rsi, 1)}</td>
        <td>${s.setup || ""}</td>
        <td class="num">${band}</td>
        <td><button class="x" data-remove="${s.ticker}" title="Remove">×</button></td>
      </tr>`;
    })
    .join("");
}

function renderDetail(s) {
  if (!s) {
    $("detail").classList.add("hidden");
    document.querySelector(".layout").classList.add("solo");
    return;
  }
  document.querySelector(".layout").classList.remove("solo");
  $("detail").classList.remove("hidden");
  $("d-symbol").textContent = s.symbol;
  $("d-name").textContent = `${s.name} · ${s.sector} · as of ${s.as_of}`;
  $("d-score").textContent = fmt(s.buy_score, 1);
  $("d-zone").className = "pill zone-pill " + s.zone;
  $("d-zone").textContent = ZONE_LABEL[s.zone] || s.zone;
  const inv = s.invalidation ? `Invalidate below <strong>${fmt(s.invalidation)}</strong>` : "";
  const inBand = s.in_buy_band ? "Price is inside the band." : "Price is outside the band.";
  const tg = s.targets || {};
  const tgt = tg.target_1
    ? `Target <strong>${fmt(tg.target_1)}</strong>${tg.target_2 ? ` → ${fmt(tg.target_2)}` : ""}${tg.source === "projected" ? " (projected, no overhead zone)" : ""}`
    : "";
  const rr = s.risk_reward != null ? `Reward:risk <strong>${fmt(s.risk_reward, 1)}</strong>` : "";
  const sizing = s.size_pct_for_1pct_risk
    ? `Stop is ${fmt(s.risk_pct, 1)}% away: ~${fmt(s.size_pct_for_1pct_risk, 0)}% of capital risks 1%`
    : "";
  $("d-band").innerHTML = s.buy_from
    ? [`Buy band <strong>${fmt(s.buy_from)} – ${fmt(s.buy_to)}</strong>`, inBand, inv, tgt, rr, sizing]
        .filter(Boolean)
        .join("<br>")
    : "No reliable demand zone to define an entry yet.";
  $("d-signals").innerHTML = (s.signals || []).map((x) => `<li>${x}</li>`).join("");
  const scores = s.scores || {};
  $("d-bars").innerHTML = ["location", "momentum", "trend", "volume", "risk"]
    .map(
      (k) => `<div>
        <div class="bar-lbl"><span>${k}</span><span>${scores[k] ?? "—"}</span></div>
        <div class="track"><span style="width:${scores[k] || 0}%"></span></div>
      </div>`
    )
    .join("");
  const t = s.technicals || {};
  const f = s.fundamentals || {};
  const bn = (s.benchmark && s.benchmark.name) || "index";
  const rows = [
    ["RSI (14)", fmt(t.rsi, 1)],
    ["ADX", fmt(t.adx, 1)],
    ["+DI / −DI", `${fmt(t.plus_di, 1)} / ${fmt(t.minus_di, 1)}`],
    ["Rel volume", fmt(t.rel_vol, 2) + "x"],
    ["CMF", fmt(t.cmf, 3)],
    [`20d vs ${bn}`, t.rs_20 == null ? "—" : (t.rs_20 > 0 ? "+" : "") + fmt(t.rs_20, 2)],
    [`60d vs ${bn}`, t.rs_60 == null ? "—" : (t.rs_60 > 0 ? "+" : "") + fmt(t.rs_60, 2)],
    ["Weekly trend", (s.weekly && s.weekly.trend) || "—"],
    [`${bn} regime`, (s.regime && s.regime.state) || "—"],
    ["Liquidity", (s.liquidity && s.liquidity.state) || "—"],
    ["60d drawdown", fmt(t.dd_60, 1) + "%"],
    ["52w position", t.pos_52w != null ? fmt(t.pos_52w, 1) + "%" : "—"],
    ["SMA 20 / dist", `${fmt(t.sma20)}  (${fmt(t.dist_sma20)}%)`],
    ["SMA 50 / dist", `${fmt(t.sma50)}  (${fmt(t.dist_sma50)}%)`],
    ["SMA 200 / dist", `${fmt(t.sma200)}  (${fmt(t.dist_sma200)}%)`],
    ["Structure", t.structure],
    ["ATR %", fmt(t.atr_pct, 2)],
    ["PE", f.pe ?? "—"],
    ["Market cap", mcap(f.market_cap)],
    ["Earnings <10d", f.earnings_soon ? "Yes" : "No"],
  ];
  $("d-tech").innerHTML = rows
    .map(([a, b]) => `<div><span>${a}</span><b>${b}</b></div>`)
    .join("");
  drawChart(s.chart || {}, s);
}

function linePath(values, w, h, padT, padB, scaleMin, scaleMax) {
  const nums = values.filter((v) => v != null);
  if (!nums.length) return { d: "", min: 0, max: 1 };
  const min = scaleMin ?? Math.min(...nums);
  const max = scaleMax ?? Math.max(...nums);
  const span = max - min || 1;
  const n = values.length;
  let d = "";
  let started = false;
  values.forEach((v, i) => {
    if (v == null) return;
    const x = (i / Math.max(n - 1, 1)) * w;
    const y = padT + (1 - (v - min) / span) * (h - padT - padB);
    d += (started ? " L " : "M ") + x.toFixed(1) + " " + y.toFixed(1);
    started = true;
  });
  return { d, min, max };
}

function zoneRects(chart, s, w, h, pMin, pMax) {
  const span = pMax - pMin || 1;
  const y = (v) => 12 + (1 - (Math.min(Math.max(v, pMin), pMax) - pMin) / span) * (h - 12 - 36);
  const rect = (lo, hi, fill, stroke) => {
    if (hi < pMin || lo > pMax) return "";
    const y1 = y(hi);
    const y2 = y(lo);
    return `<rect x="0" y="${y1.toFixed(1)}" width="${w}" height="${Math.max(y2 - y1, 1.5).toFixed(1)}" fill="${fill}" stroke="${stroke}" stroke-width="0.6"/>`;
  };
  let out = "";
  (chart.zones || []).forEach((z) => {
    out += z.role === "support"
      ? rect(z.low, z.high, "rgba(111,191,115,0.10)", "rgba(111,191,115,0.35)")
      : rect(z.low, z.high, "rgba(205,102,95,0.10)", "rgba(205,102,95,0.35)");
  });
  if (s && s.buy_from && s.buy_to) out += rect(s.buy_from, s.buy_to, "rgba(196,163,90,0.18)", "rgba(196,163,90,0.7)");
  if (s && s.invalidation && s.invalidation >= pMin) {
    const yy = y(s.invalidation).toFixed(1);
    out += `<line x1="0" y1="${yy}" x2="${w}" y2="${yy}" stroke="#cd665f" stroke-dasharray="4 3" stroke-width="0.8"/>`;
  }
  const t1 = s && s.targets && s.targets.target_1;
  if (t1 && t1 <= pMax) {
    const yy = y(t1).toFixed(1);
    out += `<line x1="0" y1="${yy}" x2="${w}" y2="${yy}" stroke="#6fbf73" stroke-dasharray="4 3" stroke-width="0.8"/>`;
  }
  return out;
}

function drawChart(chart, s) {
  const svg = $("chart");
  const rsiSvg = $("rsi-chart");
  const close = chart.close || [];
  if (!close.length) {
    svg.innerHTML = "";
    rsiSvg.innerHTML = "";
    return;
  }
  const w = 640;
  const h = 280;
  const all = close.concat(chart.sma20 || [], chart.sma50 || [], chart.sma200 || []).filter((v) => v != null);
  const pMin = Math.min(...all);
  const pMax = Math.max(...all);
  const zonesSvg = zoneRects(chart, s, w, h, pMin, pMax);
  const price = linePath(close, w, h, 12, 36, pMin, pMax);
  const s20 = linePath(chart.sma20 || [], w, h, 12, 36, pMin, pMax);
  const s50 = linePath(chart.sma50 || [], w, h, 12, 36, pMin, pMax);
  const s200 = linePath(chart.sma200 || [], w, h, 12, 36, pMin, pMax);
  const last = close.filter((v) => v != null).pop();
  svg.innerHTML = `
    ${zonesSvg}
    <path d="${price.d}" fill="none" stroke="#e8e6df" stroke-width="1.6"/>
    <path d="${s20.d}" fill="none" stroke="#c4a35a" stroke-width="1"/>
    <path d="${s50.d}" fill="none" stroke="#6f9fbf" stroke-width="1"/>
    <path d="${s200.d}" fill="none" stroke="#8b6bb0" stroke-width="1"/>
    <text x="8" y="16" fill="#8b9088" font-size="11" font-family="IBM Plex Mono">${fmt(price.max)}</text>
    <text x="8" y="${h - 8}" fill="#8b9088" font-size="11" font-family="IBM Plex Mono">${fmt(price.min)}</text>
    <text x="${w - 8}" y="16" fill="#e8e6df" font-size="11" text-anchor="end" font-family="IBM Plex Mono">${fmt(last)}</text>
    <text x="8" y="${h - 22}" fill="#c4a35a" font-size="10">SMA20</text>
    <text x="58" y="${h - 22}" fill="#6f9fbf" font-size="10">SMA50</text>
    <text x="108" y="${h - 22}" fill="#8b6bb0" font-size="10">SMA200</text>
  `;
  const rh = 70;
  const rsi = linePath(chart.rsi || [], w, rh, 8, 8);
  const y30 = 8 + (1 - (30 - rsi.min) / ((rsi.max - rsi.min) || 1)) * (rh - 16);
  const y70 = 8 + (1 - (70 - rsi.min) / ((rsi.max - rsi.min) || 1)) * (rh - 16);
  rsiSvg.innerHTML = `
    <line x1="0" y1="${y30}" x2="${w}" y2="${y30}" stroke="#2a2e36"/>
    <line x1="0" y1="${y70}" x2="${w}" y2="${y70}" stroke="#2a2e36"/>
    <path d="${rsi.d}" fill="none" stroke="#c4a35a" stroke-width="1.3"/>
    <text x="8" y="12" fill="#8b9088" font-size="10" font-family="IBM Plex Mono">RSI</text>
  `;
}

async function loadSnapshot(force) {
  setStatus("Loading prices and scoring buy zones…");
  try {
    const data = await api(force ? "/api/refresh" : "/api/snapshot", force ? { method: "POST" } : undefined);
    SNAPSHOT = data;
    renderKpis(data);
    renderTable(data);
    if (SELECTED) {
      const found = data.stocks.find((s) => s.ticker === SELECTED);
      renderDetail(found || null);
    } else if (data.stocks.length) {
      SELECTED = data.stocks[0].ticker;
      renderTable(data);
      renderDetail(data.stocks[0]);
    } else {
      renderDetail(null);
    }
    const err = (data.errors || []).length;
    setStatus(
      `${data.count} stocks scored${err ? ` · ${err} failed` : ""} · data cached 15 minutes`
    );
  } catch (e) {
    setStatus(e.message);
  }
}

$("add-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const ticker = $("ticker-input").value.trim();
  if (!ticker) return;
  setStatus(`Adding ${ticker}…`);
  try {
    const res = await api("/api/watchlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    $("ticker-input").value = "";
    SELECTED = res.snapshot?.ticker || SELECTED;
    await loadSnapshot(false);
  } catch (e) {
    setStatus(e.message);
  }
});

$("refresh").addEventListener("click", () => loadSnapshot(true));
$("close-detail").addEventListener("click", () => {
  SELECTED = null;
  renderDetail(null);
  renderTable(SNAPSHOT);
});

$("rows").addEventListener("click", async (ev) => {
  const rm = ev.target.closest("[data-remove]");
  if (rm) {
    ev.stopPropagation();
    const t = rm.getAttribute("data-remove");
    await api("/api/watchlist/" + encodeURIComponent(t), { method: "DELETE" });
    if (SELECTED === t) SELECTED = null;
    await loadSnapshot(false);
    return;
  }
  const tr = ev.target.closest("tr[data-ticker]");
  if (!tr) return;
  SELECTED = tr.getAttribute("data-ticker");
  const found = (SNAPSHOT.stocks || []).find((s) => s.ticker === SELECTED);
  renderTable(SNAPSHOT);
  renderDetail(found);
});

document.querySelector(".layout").classList.add("solo");
loadSnapshot(false);
