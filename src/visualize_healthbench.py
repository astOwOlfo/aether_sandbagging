"""Interactive, self-contained HTML visualization of evaluation scores.

Usage:
    from visualize import visualize
    visualize(scores, "report.html")

The generated file embeds all data (prompts, rubrics, every response, every
rubric judgment with its explanation) plus plotly.js, so it works offline and
nothing is dropped. Navigation: an Overview tab with aggregate scores and
distributions, a sortable/filterable per-datapoint table, and a Datapoint tab
with a dropdown for drilling down to individual responses and rubric grades.
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from statistics import mean
from typing import TYPE_CHECKING, Any

from plotly.offline import get_plotlyjs

if TYPE_CHECKING:
    from src.healthbench import DatapointScores, ResponseScore


def _is_failure(sample: Any) -> bool:
    return isinstance(sample, Enum)


def _response_score(score: "ResponseScore") -> float:
    achievable = sum(
        s.rubric.points for s in score.rubric_scores if s.rubric.points > 0
    )
    if achievable == 0:
        return 0.0
    return sum(s.rubric.points for s in score.rubric_scores if s.met) / achievable


def _sample_to_dict(sample: Any) -> dict:
    if _is_failure(sample):
        return {"failure": sample.name}
    achieved = sum(s.rubric.points for s in sample.rubric_scores if s.met)
    achievable = sum(
        s.rubric.points for s in sample.rubric_scores if s.rubric.points > 0
    )
    return {
        "score": _response_score(sample),
        "achieved_points": achieved,
        "achievable_points": achievable,
        "response": sample.response.completion,
        "reasoning": sample.response.reasoning,
        "rubric_scores": [
            {
                "criterion": rs.rubric.criterion,
                "points": rs.rubric.points,
                "met": rs.met,
                "explanation": rs.explanation,
            }
            for rs in sample.rubric_scores
        ],
    }


def _side_mean(samples: list[dict] | None) -> float | None:
    if samples is None:
        return None
    valid = [s["score"] for s in samples if "failure" not in s]
    return mean(valid) if valid else None


def _datapoint_to_dict(index: int, dp_scores: "DatapointScores") -> dict:
    malicious = [_sample_to_dict(s) for s in dp_scores.malicious_scores]
    control = (
        None
        if dp_scores.control_scores is None
        else [_sample_to_dict(s) for s in dp_scores.control_scores]
    )
    return {
        "index": index,
        "original_prompt": dp_scores.datapoint.original_prompt,
        "malicious_prompt": dp_scores.datapoint.malicious_prompt,
        "control_prompt": dp_scores.datapoint.control_prompt,
        "rubrics": [
            {"criterion": r.criterion, "points": r.points}
            for r in dp_scores.datapoint.rubrics
        ],
        "malicious": malicious,
        "control": control,
        "malicious_mean": _side_mean(malicious),
        "control_mean": _side_mean(control),
    }


def _build_summary(datapoints: list[dict]) -> dict:
    total_malicious_samples = sum(len(d["malicious"]) for d in datapoints)
    total_control_samples = sum(
        len(d["control"]) for d in datapoints if d["control"] is not None
    )

    malicious_failures: Counter[str] = Counter()
    control_failures: Counter[str] = Counter()
    for d in datapoints:
        for s in d["malicious"]:
            if "failure" in s:
                malicious_failures[s["failure"]] += 1
        if d["control"] is None:
            control_failures["missing"] += len(d["malicious"])
        else:
            for s in d["control"]:
                if "failure" in s:
                    control_failures[s["failure"]] += 1

    valid = [
        d
        for d in datapoints
        if d["malicious_mean"] is not None and d["control_mean"] is not None
    ]
    comparisons = Counter(
        (d["malicious_mean"] > d["control_mean"])
        - (d["malicious_mean"] < d["control_mean"])
        for d in valid
    )

    return {
        "n_datapoints": len(datapoints),
        "n_valid": len(valid),
        "total_malicious_samples": total_malicious_samples,
        "total_control_samples": total_control_samples,
        "average_malicious_score_when_valid": (
            mean(d["malicious_mean"] for d in valid) if valid else None
        ),
        "average_control_score_when_valid": (
            mean(d["control_mean"] for d in valid) if valid else None
        ),
        "n_malicious_underperforms": comparisons.get(-1, 0),
        "n_control_underperforms": comparisons.get(1, 0),
        "n_malicious_control_tie": comparisons.get(0, 0),
        # fractions use the same denominator convention as failure_frequencies:
        # total number of malicious samples
        "malicious_failure_fractions": {
            name: count / total_malicious_samples
            for name, count in malicious_failures.items()
        },
        "control_failure_fractions": {
            name: count / total_malicious_samples
            for name, count in control_failures.items()
        },
        "malicious_failure_counts": dict(malicious_failures),
        "control_failure_counts": dict(control_failures),
    }


def visualize(scores: list["DatapointScores"], html_filename: str) -> None:
    datapoints = [_datapoint_to_dict(i, s) for i, s in enumerate(scores)]
    data = {"datapoints": datapoints, "summary": _build_summary(datapoints)}
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    html = _PAGE_TEMPLATE.replace("__PLOTLY_JS__", get_plotlyjs()).replace(
        "__DATA_JSON__", data_json
    )
    with open(html_filename, "w") as f:
        f.write(html)


_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation scores</title>
<style>
  :root {
    --mal: #d62728;
    --ctl: #1f77b4;
    --fail: #9467bd;
    --bg: #f6f7f9;
    --card: #ffffff;
    --border: #dde1e6;
    --text: #1b1f24;
    --muted: #5b6470;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.45;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px 0 24px;
    position: sticky;
    top: 0;
    z-index: 20;
  }
  header h1 { margin: 0 0 10px 0; font-size: 18px; }
  .tabs { display: flex; gap: 4px; }
  .tab-btn {
    border: 1px solid var(--border);
    border-bottom: none;
    background: var(--bg);
    color: var(--muted);
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
    cursor: pointer;
    font-size: 14px;
  }
  .tab-btn.active { background: var(--card); color: var(--text); font-weight: 600;
    box-shadow: 0 2px 0 var(--card); }
  main { padding: 18px 24px 60px 24px; max-width: 1500px; margin: 0 auto; }
  .tab-page { display: none; }
  .tab-page.active { display: block; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    min-width: 130px;
  }
  .card .v { font-size: 22px; font-weight: 700; }
  .card .l { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .card .v.mal { color: var(--mal); }
  .card .v.ctl { color: var(--ctl); }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
  }
  .panel h2 { margin: 0 0 8px 0; font-size: 15px; }
  .panel .hint { color: var(--muted); font-size: 12px; margin: 0 0 6px 0; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 1100px) { .chart-grid { grid-template-columns: 1fr; } }
  .plot { width: 100%; }
  table.grid { border-collapse: collapse; width: 100%; }
  table.grid th, table.grid td {
    border: 1px solid var(--border);
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
  }
  table.grid th { background: var(--bg); cursor: pointer; user-select: none; white-space: nowrap; }
  table.grid th .arrow { color: var(--muted); font-size: 11px; }
  table.grid tbody tr:hover { background: #eef3fb; cursor: pointer; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .mal-text { color: var(--mal); font-weight: 600; }
  .ctl-text { color: var(--ctl); font-weight: 600; }
  .fail-text { color: var(--fail); font-weight: 600; }
  .snippet { color: var(--muted); }
  input.filter, select {
    font: inherit;
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--card);
  }
  .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  .toolbar select { max-width: 900px; flex: 1; min-width: 300px; }
  button.nav {
    font: inherit;
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--card);
    cursor: pointer;
  }
  button.nav:hover { background: var(--bg); }
  details { margin: 6px 0; }
  details > summary {
    cursor: pointer;
    padding: 7px 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-weight: 600;
  }
  details[open] > summary { border-radius: 6px 6px 0 0; }
  details > .body {
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 10px;
    background: var(--card);
  }
  .msg { margin: 6px 0; }
  .msg .role {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2px;
  }
  pre.text {
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    margin: 2px 0;
    font-family: inherit;
    max-height: 420px;
    overflow-y: auto;
  }
  .subtabs { display: flex; gap: 4px; margin: 4px 0 10px 0; }
  .subtab-btn {
    border: 1px solid var(--border);
    background: var(--bg);
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
  }
  .subtab-btn.active { background: var(--card); font-weight: 700; }
  .subtab-btn.mal.active { color: var(--mal); border-color: var(--mal); }
  .subtab-btn.ctl.active { color: var(--ctl); border-color: var(--ctl); }
  .met-yes { color: #1a7f37; font-weight: 700; }
  .met-no { color: #b30000; font-weight: 700; }
  .pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
    background: var(--bg);
    border: 1px solid var(--border);
  }
</style>
</head>
<body>
<script>__PLOTLY_JS__</script>
<script id="data" type="application/json">__DATA_JSON__</script>

<header>
  <h1>Evaluation scores — malicious vs control</h1>
  <div class="tabs">
    <button class="tab-btn active" data-tab="overview">Overview</button>
    <button class="tab-btn" data-tab="table">Datapoint table</button>
    <button class="tab-btn" data-tab="detail">Datapoint detail</button>
  </div>
</header>

<main>
  <div id="tab-overview" class="tab-page active">
    <div class="cards" id="stat-cards"></div>
    <div class="chart-grid">
      <div class="panel"><h2>Per-response score distribution</h2>
        <p class="hint">Every valid (non-failed) sampled response, all datapoints.</p>
        <div id="plot-resp-hist" class="plot"></div></div>
      <div class="panel"><h2>Per-datapoint mean score distribution</h2>
        <p class="hint">Mean over valid samples of each datapoint.</p>
        <div id="plot-dp-hist" class="plot"></div></div>
      <div class="panel"><h2>Malicious vs control per datapoint</h2>
        <p class="hint">Each point is a datapoint with valid samples on both sides; below the diagonal = malicious underperforms. Click a point to open its detail view.</p>
        <div id="plot-scatter" class="plot"></div></div>
      <div class="panel"><h2>Score difference (malicious − control) per datapoint</h2>
        <p class="hint">Negative = malicious framing scores worse.</p>
        <div id="plot-delta-hist" class="plot"></div></div>
      <div class="panel"><h2>Failure fractions</h2>
        <p class="hint">Fraction of all malicious samples (same denominator for both sides; "missing" = control never sampled because all malicious samples failed).</p>
        <div id="plot-failures" class="plot"></div></div>
      <div class="panel"><h2>Refusals per datapoint</h2>
        <p class="hint">Number of REFUSED samples within each datapoint.</p>
        <div id="plot-refusals" class="plot"></div></div>
    </div>
  </div>

  <div id="tab-table" class="tab-page">
    <div class="panel">
      <div class="toolbar">
        <input id="table-filter" class="filter" type="search"
               placeholder="Filter by prompt text…" size="40">
        <span class="hint" id="table-count"></span>
      </div>
      <p class="hint">Click a column header to sort, a row to open the datapoint's detail view.</p>
      <table class="grid" id="dp-table">
        <thead><tr id="dp-table-head"></tr></thead>
        <tbody id="dp-table-body"></tbody>
      </table>
    </div>
  </div>

  <div id="tab-detail" class="tab-page">
    <div class="toolbar">
      <button class="nav" id="prev-dp">◀ prev</button>
      <select id="dp-select"></select>
      <button class="nav" id="next-dp">next ▶</button>
    </div>
    <div id="detail-content"></div>
  </div>
</main>

<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const DPS = DATA.datapoints;
const SUMMARY = DATA.summary;
const MAL = getComputedStyle(document.documentElement).getPropertyValue("--mal").trim();
const CTL = getComputedStyle(document.documentElement).getPropertyValue("--ctl").trim();
const FAIL = getComputedStyle(document.documentElement).getPropertyValue("--fail").trim();
const PLOT_CONFIG = {displaylogo: false, responsive: true};

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function fmt(x, digits = 3) {
  if (x === null || x === undefined) return "—";
  return Number(x).toFixed(digits);
}
function pct(x) {
  if (x === null || x === undefined) return "—";
  return (100 * x).toFixed(1) + "%";
}
function validSamples(samples) {
  return (samples || []).filter(s => !("failure" in s));
}
function promptText(prompt) {
  if (typeof prompt === "string") return prompt;
  return prompt.map(m => m.content).join("\n");
}
function snippet(text, n = 110) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > n ? t.slice(0, n) + "…" : t;
}
function nRefused(samples) {
  return (samples || []).filter(s => s.failure === "REFUSED").length;
}
function nFailed(samples) {
  return (samples || []).filter(s => "failure" in s).length;
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => showTab(btn.dataset.tab));
});
function showTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-page").forEach(p =>
    p.classList.toggle("active", p.id === "tab-" + name));
  window.dispatchEvent(new Event("resize")); // let plotly reflow hidden plots
}

/* ---------- overview ---------- */
function statCards() {
  const s = SUMMARY;
  const cards = [
    [s.n_datapoints, "datapoints"],
    [s.n_valid, "valid on both sides"],
    [s.total_malicious_samples, "malicious samples"],
    [s.total_control_samples, "control samples"],
    [fmt(s.average_malicious_score_when_valid), "avg malicious score (valid)", "mal"],
    [fmt(s.average_control_score_when_valid), "avg control score (valid)", "ctl"],
    [s.n_malicious_underperforms, "malicious < control"],
    [s.n_control_underperforms, "malicious > control"],
    [s.n_malicious_control_tie, "ties"],
  ];
  document.getElementById("stat-cards").innerHTML = cards.map(([v, l, cls]) =>
    `<div class="card"><div class="v ${cls || ""}">${esc(v)}</div>` +
    `<div class="l">${esc(l)}</div></div>`).join("");
}

function baseLayout(extra) {
  return Object.assign({
    margin: {l: 55, r: 15, t: 10, b: 45},
    height: 330,
    barmode: "overlay",
    legend: {orientation: "h", y: 1.12},
    font: {family: "system-ui, sans-serif", size: 12},
  }, extra || {});
}

function overviewPlots() {
  const malResp = [], ctlResp = [];
  DPS.forEach(d => {
    validSamples(d.malicious).forEach(s => malResp.push(s.score));
    validSamples(d.control).forEach(s => ctlResp.push(s.score));
  });
  Plotly.newPlot("plot-resp-hist", [
    {x: malResp, type: "histogram", name: "malicious", opacity: 0.6,
     marker: {color: MAL}, xbins: {size: 0.05}},
    {x: ctlResp, type: "histogram", name: "control", opacity: 0.6,
     marker: {color: CTL}, xbins: {size: 0.05}},
  ], baseLayout({xaxis: {title: {text: "response score"}},
                 yaxis: {title: {text: "responses"}}}), PLOT_CONFIG);

  const malMeans = DPS.map(d => d.malicious_mean).filter(x => x !== null);
  const ctlMeans = DPS.map(d => d.control_mean).filter(x => x !== null);
  Plotly.newPlot("plot-dp-hist", [
    {x: malMeans, type: "histogram", name: "malicious", opacity: 0.6,
     marker: {color: MAL}, xbins: {size: 0.05}},
    {x: ctlMeans, type: "histogram", name: "control", opacity: 0.6,
     marker: {color: CTL}, xbins: {size: 0.05}},
  ], baseLayout({xaxis: {title: {text: "datapoint mean score"}},
                 yaxis: {title: {text: "datapoints"}}}), PLOT_CONFIG);

  const both = DPS.filter(d => d.malicious_mean !== null && d.control_mean !== null);
  const scatterDiv = document.getElementById("plot-scatter");
  Plotly.newPlot(scatterDiv, [
    {x: both.map(d => d.control_mean), y: both.map(d => d.malicious_mean),
     mode: "markers", type: "scatter", name: "datapoint",
     marker: {color: both.map(d => d.malicious_mean - d.control_mean),
              colorscale: "RdBu", cmid: 0, size: 7,
              line: {width: 0.5, color: "#666"}},
     customdata: both.map(d => d.index),
     text: both.map(d => "#" + d.index + " " + snippet(promptText(d.malicious_prompt), 70)),
     hovertemplate: "%{text}<br>control %{x:.3f} · malicious %{y:.3f}<extra></extra>"},
  ], baseLayout({
    height: 380,
    xaxis: {title: {text: "control mean score"}, range: [-0.05, 1.05]},
    yaxis: {title: {text: "malicious mean score"}, range: [-0.05, 1.05]},
    shapes: [{type: "line", x0: 0, y0: 0, x1: 1, y1: 1,
              line: {color: "#999", dash: "dot"}}],
    showlegend: false,
  }), PLOT_CONFIG);
  scatterDiv.on("plotly_click", ev => {
    openDatapoint(ev.points[0].customdata);
    showTab("detail");
  });

  const deltas = both.map(d => d.malicious_mean - d.control_mean);
  Plotly.newPlot("plot-delta-hist", [
    {x: deltas, type: "histogram", name: "malicious − control",
     marker: {color: "#7f7f7f"}, xbins: {size: 0.05}},
  ], baseLayout({
    xaxis: {title: {text: "malicious mean − control mean"}},
    yaxis: {title: {text: "datapoints"}},
    shapes: [{type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper",
              line: {color: "#333", dash: "dot"}}],
    showlegend: false,
  }), PLOT_CONFIG);

  const failNames = Array.from(new Set([
    ...Object.keys(SUMMARY.malicious_failure_fractions),
    ...Object.keys(SUMMARY.control_failure_fractions),
  ])).sort();
  Plotly.newPlot("plot-failures", [
    {x: failNames, y: failNames.map(n => SUMMARY.malicious_failure_fractions[n] || 0),
     type: "bar", name: "malicious", marker: {color: MAL},
     text: failNames.map(n => SUMMARY.malicious_failure_counts[n] || 0),
     hovertemplate: "%{x}: %{y:.1%} (%{text} samples)<extra>malicious</extra>"},
    {x: failNames, y: failNames.map(n => SUMMARY.control_failure_fractions[n] || 0),
     type: "bar", name: "control", marker: {color: CTL},
     text: failNames.map(n => SUMMARY.control_failure_counts[n] || 0),
     hovertemplate: "%{x}: %{y:.1%} (%{text} samples)<extra>control</extra>"},
  ], baseLayout({barmode: "group", yaxis: {title: {text: "fraction of samples"},
                 tickformat: ".0%"}}), PLOT_CONFIG);

  const malRefusals = DPS.map(d => nRefused(d.malicious));
  const ctlRefusals = DPS.filter(d => d.control !== null).map(d => nRefused(d.control));
  const counts = xs => {
    const c = {};
    xs.forEach(x => { c[x] = (c[x] || 0) + 1; });
    return c;
  };
  const malC = counts(malRefusals), ctlC = counts(ctlRefusals);
  const ks = Array.from(new Set([...Object.keys(malC), ...Object.keys(ctlC)]))
    .map(Number).sort((a, b) => a - b);
  Plotly.newPlot("plot-refusals", [
    {x: ks, y: ks.map(k => malC[k] || 0), type: "bar", name: "malicious",
     marker: {color: MAL}},
    {x: ks, y: ks.map(k => ctlC[k] || 0), type: "bar", name: "control",
     marker: {color: CTL}},
  ], baseLayout({barmode: "group",
                 xaxis: {title: {text: "refused samples in datapoint"}, dtick: 1},
                 yaxis: {title: {text: "datapoints"}}}), PLOT_CONFIG);
}

/* ---------- datapoint table ---------- */
const TABLE_COLS = [
  {key: "index", label: "#", num: true, get: d => d.index},
  {key: "prompt", label: "malicious prompt", num: false,
   get: d => promptText(d.malicious_prompt)},
  {key: "mal", label: "mal mean", num: true, get: d => d.malicious_mean},
  {key: "ctl", label: "ctl mean", num: true, get: d => d.control_mean},
  {key: "delta", label: "mal − ctl", num: true,
   get: d => (d.malicious_mean === null || d.control_mean === null)
     ? null : d.malicious_mean - d.control_mean},
  {key: "malvalid", label: "mal valid", num: true,
   get: d => validSamples(d.malicious).length},
  {key: "ctlvalid", label: "ctl valid", num: true,
   get: d => d.control === null ? null : validSamples(d.control).length},
  {key: "malref", label: "mal refusals", num: true, get: d => nRefused(d.malicious)},
  {key: "ctlref", label: "ctl refusals", num: true,
   get: d => d.control === null ? null : nRefused(d.control)},
];
let sortKey = "delta", sortDir = 1;

function renderTableHead() {
  document.getElementById("dp-table-head").innerHTML = TABLE_COLS.map(c => {
    const arrow = c.key === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th class="${c.num ? "num" : ""}" data-key="${c.key}">` +
      `${esc(c.label)}<span class="arrow">${arrow}</span></th>`;
  }).join("");
  document.querySelectorAll("#dp-table-head th").forEach(th => {
    th.addEventListener("click", () => {
      if (sortKey === th.dataset.key) sortDir = -sortDir;
      else { sortKey = th.dataset.key; sortDir = 1; }
      renderTableHead();
      renderTableBody();
    });
  });
}

function renderTableBody() {
  const filter = document.getElementById("table-filter").value.toLowerCase();
  const col = TABLE_COLS.find(c => c.key === sortKey);
  let rows = DPS.filter(d =>
    !filter ||
    promptText(d.malicious_prompt).toLowerCase().includes(filter) ||
    promptText(d.control_prompt).toLowerCase().includes(filter) ||
    promptText(d.original_prompt).toLowerCase().includes(filter));
  rows = rows.slice().sort((a, b) => {
    const va = col.get(a), vb = col.get(b);
    if (va === null && vb === null) return 0;
    if (va === null) return 1;  // nulls last regardless of direction
    if (vb === null) return -1;
    if (typeof va === "string") return sortDir * va.localeCompare(vb);
    return sortDir * (va - vb);
  });
  document.getElementById("table-count").textContent =
    rows.length + " / " + DPS.length + " datapoints";
  document.getElementById("dp-table-body").innerHTML = rows.map(d => {
    const cells = TABLE_COLS.map(c => {
      const v = c.get(d);
      if (c.key === "prompt")
        return `<td class="snippet">${esc(snippet(v))}</td>`;
      if (c.key === "index") return `<td class="num">${v}</td>`;
      if (v === null) return `<td class="num">—</td>`;
      if (c.key === "mal") return `<td class="num mal-text">${fmt(v)}</td>`;
      if (c.key === "ctl") return `<td class="num ctl-text">${fmt(v)}</td>`;
      if (c.key === "delta") return `<td class="num">${fmt(v)}</td>`;
      return `<td class="num">${Number.isInteger(v) ? v : fmt(v)}</td>`;
    }).join("");
    return `<tr data-index="${d.index}">${cells}</tr>`;
  }).join("");
  document.querySelectorAll("#dp-table-body tr").forEach(tr => {
    tr.addEventListener("click", () => {
      openDatapoint(Number(tr.dataset.index));
      showTab("detail");
    });
  });
}

/* ---------- datapoint detail ---------- */
let currentIndex = 0;
let currentSide = "malicious";

function dpLabel(d) {
  return `#${d.index} · mal ${fmt(d.malicious_mean)} · ctl ${fmt(d.control_mean)}` +
    ` · ${snippet(promptText(d.malicious_prompt), 90)}`;
}

function buildSelect() {
  const sel = document.getElementById("dp-select");
  sel.innerHTML = DPS.map(d =>
    `<option value="${d.index}">${esc(dpLabel(d))}</option>`).join("");
  sel.addEventListener("change", () => openDatapoint(Number(sel.value)));
  document.getElementById("prev-dp").addEventListener("click", () =>
    openDatapoint((currentIndex - 1 + DPS.length) % DPS.length));
  document.getElementById("next-dp").addEventListener("click", () =>
    openDatapoint((currentIndex + 1) % DPS.length));
}

function conversationHtml(prompt) {
  const messages = typeof prompt === "string"
    ? [{role: "user", content: prompt}] : prompt;
  return messages.map(m =>
    `<div class="msg"><span class="role">${esc(m.role)}</span>` +
    `<pre class="text">${esc(m.content)}</pre></div>`).join("");
}

function promptsHtml(d) {
  const sections = [
    ["Malicious prompt", d.malicious_prompt, true],
    ["Control prompt", d.control_prompt, false],
    ["Original prompt", d.original_prompt, false],
  ];
  return sections.map(([title, prompt, open]) =>
    `<details ${open ? "open" : ""}><summary>${esc(title)}</summary>` +
    `<div class="body">${conversationHtml(prompt)}</div></details>`).join("");
}

function rubricAggregates(d) {
  // aggregate by criterion from the samples themselves, so nothing is assumed
  // about ordering; d.rubrics fixes the display order
  const agg = side => {
    const valid = validSamples(d[side]);
    const map = {};
    valid.forEach(s => s.rubric_scores.forEach(rs => {
      const m = map[rs.criterion] || (map[rs.criterion] = {met: 0, total: 0});
      m.total += 1;
      if (rs.met) m.met += 1;
    }));
    return map;
  };
  const malAgg = agg("malicious");
  const ctlAgg = d.control === null ? {} : agg("control");
  return d.rubrics.map(r => {
    const m = malAgg[r.criterion], c = ctlAgg[r.criterion];
    return {
      criterion: r.criterion,
      points: r.points,
      malFrac: m && m.total ? m.met / m.total : null,
      ctlFrac: c && c.total ? c.met / c.total : null,
    };
  });
}

function sampleSummaryLine(s, i) {
  if ("failure" in s)
    return `sample ${i} — <span class="fail-text">${esc(s.failure)}</span>`;
  return `sample ${i} — score ${fmt(s.score)} ` +
    `<span class="pill">${s.achieved_points} / ${s.achievable_points} pts</span>`;
}

function sampleHtml(s, i) {
  let body;
  if ("failure" in s) {
    body = `<p class="hint">This sample failed with <b>${esc(s.failure)}</b>; ` +
      `no response was graded.</p>`;
  } else {
    const rows = s.rubric_scores.map(rs =>
      `<tr><td>${rs.met
          ? '<span class="met-yes">✓ met</span>'
          : '<span class="met-no">✗ not met</span>'}</td>` +
      `<td class="num">${rs.points}</td>` +
      `<td>${esc(rs.criterion)}</td>` +
      `<td class="snippet">${rs.explanation === null ? "—" : esc(rs.explanation)}</td></tr>`
    ).join("");
    const cot = s.reasoning === null || s.reasoning === undefined
      ? ""
      : `<details><summary>chain of thought</summary><div class="body">` +
        `<pre class="text">${esc(s.reasoning)}</pre></div></details>`;
    body = cot +
      `<div class="msg"><span class="role">response</span>` +
      `<pre class="text">${esc(s.response)}</pre></div>` +
      `<table class="grid"><thead><tr><th>met</th><th class="num">points</th>` +
      `<th>criterion</th><th>grader explanation</th></tr></thead>` +
      `<tbody>${rows}</tbody></table>`;
  }
  return `<details><summary>${sampleSummaryLine(s, i)}</summary>` +
    `<div class="body">${body}</div></details>`;
}

function renderSamples(d) {
  const container = document.getElementById("samples-list");
  const samples = d[currentSide];
  document.querySelectorAll(".subtab-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.side === currentSide));
  if (samples === null) {
    container.innerHTML = `<p class="hint">Control was never sampled for this ` +
      `datapoint because all malicious samples failed.</p>`;
    return;
  }
  container.innerHTML = samples.map((s, i) => sampleHtml(s, i)).join("");
}

function detailPlots(d) {
  const sideTrace = (samples, name, color, offset) => ({
    x: samples.map((_, i) => i),
    y: samples.map(s => "failure" in s ? 0 : s.score),
    type: "bar", name, marker: {
      color: samples.map(s => "failure" in s ? FAIL : color),
      opacity: samples.map(s => "failure" in s ? 0.45 : 1),
    },
    customdata: samples.map(s => "failure" in s ? s.failure : "score " + fmt(s.score)),
    hovertemplate: "sample %{x}: %{customdata}<extra>" + name + "</extra>",
  });
  const traces = [sideTrace(d.malicious, "malicious", MAL)];
  if (d.control !== null) traces.push(sideTrace(d.control, "control", CTL));
  Plotly.react("plot-samples", traces, baseLayout({
    barmode: "group", height: 280,
    xaxis: {title: {text: "sample"}, dtick: 1},
    yaxis: {title: {text: "score"}, range: [-0.05, 1.05]},
  }), PLOT_CONFIG);

  const rubrics = rubricAggregates(d);
  const labels = rubrics.map((r, i) => `R${i} (${r.points > 0 ? "+" : ""}${r.points})`);
  const hover = rubrics.map(r => snippet(r.criterion, 160));
  Plotly.react("plot-rubrics", [
    {y: labels, x: rubrics.map(r => r.malFrac), type: "bar", orientation: "h",
     name: "malicious", marker: {color: MAL}, text: hover,
     hovertemplate: "%{text}<br>met in %{x:.0%} of valid samples<extra>malicious</extra>"},
    {y: labels, x: rubrics.map(r => r.ctlFrac), type: "bar", orientation: "h",
     name: "control", marker: {color: CTL}, text: hover,
     hovertemplate: "%{text}<br>met in %{x:.0%} of valid samples<extra>control</extra>"},
  ], baseLayout({
    barmode: "group",
    height: Math.max(240, 28 * rubrics.length + 90),
    margin: {l: 90, r: 15, t: 10, b: 45},
    xaxis: {title: {text: "fraction of valid samples meeting criterion"},
            tickformat: ".0%", range: [0, 1]},
    yaxis: {autorange: "reversed"},
  }), PLOT_CONFIG);
}

function rubricTableHtml(d) {
  const rubrics = rubricAggregates(d);
  const rows = rubrics.map((r, i) =>
    `<tr><td class="num">R${i}</td><td class="num">${r.points}</td>` +
    `<td>${esc(r.criterion)}</td>` +
    `<td class="num mal-text">${pct(r.malFrac)}</td>` +
    `<td class="num ctl-text">${pct(r.ctlFrac)}</td></tr>`).join("");
  return `<table class="grid"><thead><tr><th class="num">#</th>` +
    `<th class="num">points</th><th>criterion</th>` +
    `<th class="num">mal met</th><th class="num">ctl met</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>`;
}

function openDatapoint(index) {
  currentIndex = index;
  const d = DPS[index];
  document.getElementById("dp-select").value = String(index);

  const delta = (d.malicious_mean === null || d.control_mean === null)
    ? null : d.malicious_mean - d.control_mean;
  const nMal = d.malicious.length;
  const nCtl = d.control === null ? 0 : d.control.length;
  const cards = [
    [fmt(d.malicious_mean), "malicious mean", "mal"],
    [fmt(d.control_mean), "control mean", "ctl"],
    [fmt(delta), "mal − ctl"],
    [`${validSamples(d.malicious).length} / ${nMal}`, "malicious valid"],
    [d.control === null ? "—" : `${validSamples(d.control).length} / ${nCtl}`,
     "control valid"],
    [nRefused(d.malicious), "malicious refusals"],
    [d.control === null ? "—" : nRefused(d.control), "control refusals"],
    [d.rubrics.length, "rubrics"],
  ];

  document.getElementById("detail-content").innerHTML =
    `<div class="cards">${cards.map(([v, l, cls]) =>
      `<div class="card"><div class="v ${cls || ""}">${esc(v)}</div>` +
      `<div class="l">${esc(l)}</div></div>`).join("")}</div>` +
    `<div class="panel"><h2>Prompts</h2>${promptsHtml(d)}</div>` +
    `<div class="panel"><h2>Per-sample scores</h2>` +
    `<p class="hint">Purple bars at 0 are failed samples (hover for the failure kind).</p>` +
    `<div id="plot-samples" class="plot"></div></div>` +
    `<div class="panel"><h2>Rubric criteria</h2>` +
    `<div id="plot-rubrics" class="plot"></div>${rubricTableHtml(d)}</div>` +
    `<div class="panel"><h2>Individual samples</h2>` +
    `<div class="subtabs">` +
    `<button class="subtab-btn mal" data-side="malicious">Malicious (${nMal})</button>` +
    `<button class="subtab-btn ctl" data-side="control">Control (${nCtl})</button>` +
    `</div><div id="samples-list"></div></div>`;

  document.querySelectorAll(".subtab-btn").forEach(btn =>
    btn.addEventListener("click", () => {
      currentSide = btn.dataset.side;
      renderSamples(d);
    }));
  renderSamples(d);
  detailPlots(d);
}

/* ---------- init ---------- */
statCards();
overviewPlots();
renderTableHead();
renderTableBody();
document.getElementById("table-filter").addEventListener("input", renderTableBody);
buildSelect();
if (DPS.length > 0) openDatapoint(0);
</script>
</body>
</html>
"""
