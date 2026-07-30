"""Browsable HTML report of every completion in a healthbench experiment.

Usage:
    from src.visualize_healthbench import visualize
    visualize(
        models, results, title, healthbench_grader, refusal_judge,
        statement_correctness_judge, "visualizations/healthbench.html",
    )

One tab per model. Each tab shows summary stat tiles, distributions (scores,
response outcomes, paired malicious-minus-control score differences), and a
filterable, paginated browser over every datapoint with the full malicious and
control completions (including refusals), reasoning, per-rubric grades, and the
contradicting claims extracted from each response pair.

The file is self-contained: plotly.js is embedded, and the datapoint browser is
stored as JSON and rendered client-side a page at a time so the page stays
responsive with thousands of completions.
"""

import json
from html import escape
from statistics import fmean

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from src.healthbench import (
    DatapointScores,
    EvaluatedContradictingClaims,
    ExperimentResult,
    Failure,
    Refusal,
    ResponseScore,
    Rubric,
    conversation_to_str,
)
from src.llm_apis import Completion, Model

_CONTROL_COLOR = "#2a78d6"
_MALICIOUS_COLOR = "#eb6834"
_DIFF_COLOR = "#1baf7a"

_FONT = {
    "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
    "color": "#0b0b0b",
    "size": 12,
}

_CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #f9f9f7; color: #0b0b0b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 17px; }
.caption { color: #52514e; font-size: 13px; margin-top: 0; }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 18px 0; }
.tab-button { border: 1px solid #c3c2b7; background: #fcfcfb; color: #0b0b0b;
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font: inherit; font-size: 14px; }
.tab-button.active { background: #2a78d6; border-color: #2a78d6; color: #ffffff; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.tile { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 8px; padding: 10px 16px; min-width: 140px; }
.tile .value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .label { font-size: 12px; color: #52514e; margin-top: 2px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 16px; margin-bottom: 28px; }
.chart-box { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 8px; padding: 10px 12px 4px; }
.chart-box h3 { margin: 2px 0 6px; font-size: 14px; }
.chart { width: 100%; height: 300px; }
.browser-controls { display: flex; flex-wrap: wrap; align-items: center;
  gap: 12px; margin: 10px 0 14px; font-size: 13px; }
select, .page-button { font: inherit; font-size: 13px; padding: 4px 8px; }
.page-info { color: #52514e; }
details.datapoint { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 8px; margin-bottom: 10px; padding: 4px 14px; }
details.datapoint > summary { cursor: pointer; padding: 8px 0; font-size: 14px;
  line-height: 1.5; }
.snippet { color: #52514e; }
.summary-stats { color: #52514e; font-size: 12.5px; }
.badge { display: inline-block; border-radius: 10px; padding: 1px 8px;
  font-size: 11.5px; border: 1px solid; margin-left: 6px; vertical-align: middle;
  background: #fcfcfb; }
.badge-scored { color: #006300; border-color: #0ca30c; }
.badge-refused { color: #9c2a2a; border-color: #d03b3b; }
.badge-failure { color: #8a5a00; border-color: #fab219; }
.badge-missing { color: #52514e; border-color: #c3c2b7; }
.pair { border-top: 1px solid #e1e0d9; padding: 8px 0; }
.pair h4 { margin: 4px 0 8px; font-size: 13px; color: #52514e; }
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 900px) { .columns { grid-template-columns: 1fr; } }
.response { border-left: 3px solid; border-radius: 4px; padding: 4px 10px 8px;
  background: #f9f9f7; min-width: 0; }
.response.malicious { border-left-color: #eb6834; }
.response.control { border-left-color: #2a78d6; }
.response h5 { margin: 4px 0 6px; font-size: 13px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-family: inherit;
  font-size: 13px; line-height: 1.45; background: #fcfcfb;
  border: 1px solid #e1e0d9; border-radius: 6px; padding: 8px 10px; margin: 6px 0; }
details.sub { margin: 6px 0; font-size: 13px; }
details.sub > summary { cursor: pointer; color: #1c5cab; }
.message .role { font-size: 11.5px; font-weight: 600; text-transform: uppercase;
  color: #52514e; }
ul.rubrics { list-style: none; padding-left: 0; font-size: 13px; margin: 6px 0; }
ul.rubrics li { margin: 6px 0; padding-left: 10px; border-left: 2px solid #e1e0d9; }
.points { color: #52514e; font-variant-numeric: tabular-nums; margin-right: 4px; }
.explanation { color: #52514e; margin-top: 2px; }
.mark-met { color: #006300; font-weight: 600; }
.mark-unmet { color: #9c2a2a; font-weight: 600; }
ul.claims-list { list-style: none; padding-left: 0; font-size: 13px; margin: 6px 0; }
ul.claims-list li { margin: 8px 0; padding-left: 10px; border-left: 2px solid #e1e0d9; }
.claims { margin-top: 8px; }
.note { color: #52514e; font-size: 13px; }
.empty { color: #52514e; }
"""

_JS = """
const DATA = JSON.parse(document.getElementById("viz-data").textContent);
const PAGE_SIZE = 20;
const state = DATA.models.map(function () { return { page: 0, filter: "all" }; });

function filteredIndices(m) {
  const filter = state[m].filter;
  const indices = [];
  DATA.models[m].datapoints.forEach(function (d, i) {
    if (filter === "refusal" && d.refusals === 0) return;
    if (filter === "failure" && d.failures === 0) return;
    indices.push(i);
  });
  return indices;
}

function renderBrowser(m) {
  const panel = document.getElementById("panel-" + m);
  const indices = filteredIndices(m);
  const pages = Math.max(1, Math.ceil(indices.length / PAGE_SIZE));
  state[m].page = Math.min(Math.max(state[m].page, 0), pages - 1);
  const start = state[m].page * PAGE_SIZE;
  const shown = indices.slice(start, start + PAGE_SIZE);
  panel.querySelector(".datapoint-list").innerHTML = shown.length
    ? shown.map(function (i) { return DATA.models[m].datapoints[i].html; }).join("")
    : '<p class="empty">No datapoints match this filter.</p>';
  panel.querySelector(".page-info").textContent =
    "page " + (state[m].page + 1) + " of " + pages + " \\u2014 " +
    indices.length + " datapoint" + (indices.length === 1 ? "" : "s");
}

function renderCharts(m) {
  const panel = document.getElementById("panel-" + m);
  panel.querySelectorAll(".chart").forEach(function (div) {
    if (div.dataset.rendered) return;
    const fig = DATA.models[m].figures[div.dataset.fig];
    Plotly.newPlot(div, fig.data, fig.layout, { displayModeBar: false, responsive: true });
    div.dataset.rendered = "1";
  });
}

function showTab(m) {
  document.querySelectorAll(".tab-button").forEach(function (button, i) {
    button.classList.toggle("active", i === m);
  });
  document.querySelectorAll(".model-panel").forEach(function (panel, i) {
    panel.hidden = i !== m;
  });
  renderCharts(m);
}

document.querySelectorAll(".tab-button").forEach(function (button, m) {
  button.addEventListener("click", function () { showTab(m); });
});
document.querySelectorAll(".model-panel").forEach(function (panel, m) {
  panel.querySelector(".filter-select").addEventListener("change", function (event) {
    state[m].filter = event.target.value;
    state[m].page = 0;
    renderBrowser(m);
  });
  panel.querySelector(".prev-page").addEventListener("click", function () {
    state[m].page -= 1;
    renderBrowser(m);
  });
  panel.querySelector(".next-page").addEventListener("click", function () {
    state[m].page += 1;
    renderBrowser(m);
  });
  renderBrowser(m);
});
if (DATA.models.length > 0) showTab(0);
"""


def _esc(text: str) -> str:
    return escape(text)


def _labels(models: list[Model]) -> list[str]:
    labels: list[str] = [model.model.split("/")[-1] for model in models]
    if len(set(labels)) != len(labels):
        labels = [model.model for model in models]
    return labels


def _response_score(score: ResponseScore) -> float:
    return sum(s.rubric.points for s in score.rubric_scores if s.met) / sum(
        s.rubric.points for s in score.rubric_scores if s.rubric.points > 0
    )


def _fmt_mean(values: list[float]) -> str:
    if not values:
        return "–"
    return f"{fmean(values):.2f}"


def _failure_label(failure: Failure) -> str:
    return failure.name.replace("_", " ").lower()


def _snippet(prompt: str | list[dict], limit: int = 120) -> str:
    text: str = prompt if isinstance(prompt, str) else conversation_to_str(prompt)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _badge(kind: str, text: str) -> str:
    return f'<span class="badge badge-{kind}">{_esc(text)}</span>'


def _prompt_block(title: str, prompt: str | list[dict]) -> str:
    if isinstance(prompt, str):
        body = f"<pre>{_esc(prompt)}</pre>"
    else:
        body = "".join(
            f'<div class="message"><span class="role">{_esc(message["role"])}</span>'
            f"<pre>{_esc(message['content'])}</pre></div>"
            for message in prompt
        )
    return f'<details class="sub"><summary>{_esc(title)}</summary>{body}</details>'


def _completion_block(response: Completion) -> str:
    parts: list[str] = []
    if response.reasoning is not None:
        parts.append(
            '<details class="sub"><summary>Reasoning</summary>'
            f"<pre>{_esc(response.reasoning)}</pre></details>"
        )
    parts.append(f"<pre>{_esc(response.completion)}</pre>")
    return "".join(parts)


def _rubrics_block(rubrics: list[Rubric]) -> str:
    items = "".join(
        f'<li><span class="points">[{rubric.points:g}]</span>{_esc(rubric.criterion)}</li>'
        for rubric in rubrics
    )
    return (
        f'<details class="sub"><summary>Rubrics ({len(rubrics)})</summary>'
        f'<ul class="rubrics">{items}</ul></details>'
    )


def _rubric_grades_block(score: ResponseScore) -> str:
    items: list[str] = []
    for rubric_score in score.rubric_scores:
        mark = (
            '<span class="mark-met">✓ met</span>'
            if rubric_score.met
            else '<span class="mark-unmet">✗ not met</span>'
        )
        explanation = (
            f'<div class="explanation">{_esc(rubric_score.explanation)}</div>'
            if rubric_score.explanation
            else ""
        )
        items.append(
            f'<li><span class="points">[{rubric_score.rubric.points:g}]</span>'
            f"{mark} — {_esc(rubric_score.rubric.criterion)}{explanation}</li>"
        )
    return (
        f'<details class="sub"><summary>Per-rubric grades '
        f"(score {_response_score(score):.2f})</summary>"
        f'<ul class="rubrics">{"".join(items)}</ul></details>'
    )


def _response_card(side: str, score: ResponseScore | Refusal | Failure | None) -> str:
    if score is None:
        badge = _badge("missing", "not generated")
        body = '<p class="note">Control responses were not generated because every malicious response failed.</p>'
    elif isinstance(score, ResponseScore):
        badge = _badge("scored", f"score {_response_score(score):.2f}")
        body = _completion_block(score.response) + _rubric_grades_block(score)
    elif isinstance(score, Refusal):
        badge = _badge("refused", "excluded")
        if score.response is None:
            body = '<p class="note">Excluded via provider content filter — no completion was returned.</p>'
        else:
            body = _completion_block(score.response)
    else:
        badge = _badge("failure", f"failure: {_failure_label(score)}")
        body = '<p class="note">No usable evaluation for this response.</p>'
    return f'<div class="response {side}"><h5>{side}{badge}</h5>{body}</div>'


def _claims_block(claims: list[EvaluatedContradictingClaims] | Failure) -> str:
    if isinstance(claims, Failure):
        return (
            f'<div class="claims note">Contradicting-claim extraction failed: '
            f"{_esc(_failure_label(claims))}</div>"
        )
    if not claims:
        return '<div class="claims note">No contradicting claims between this response pair.</div>'
    items: list[str] = []
    for claim in claims:
        mal_mark = (
            '<span class="mark-met">judged correct</span>'
            if claim.malicious_correct
            else '<span class="mark-unmet">judged incorrect</span>'
        )
        ctrl_mark = (
            '<span class="mark-met">judged correct</span>'
            if claim.control_correct
            else '<span class="mark-unmet">judged incorrect</span>'
        )
        items.append(
            f"<li><div>malicious ({mal_mark}): {_esc(claim.malicious_claim)}</div>"
            f"<div>control ({ctrl_mark}): {_esc(claim.control_claim)}</div></li>"
        )
    return (
        f'<div class="claims"><details class="sub">'
        f"<summary>Contradicting claims ({len(claims)})</summary>"
        f'<ul class="claims-list">{"".join(items)}</ul></details></div>'
    )


def _datapoint_html(index: int, datapoint_scores: DatapointScores) -> dict:
    malicious = datapoint_scores.malicious_scores
    control = datapoint_scores.control_scores

    all_responses = list(malicious) + (list(control) if control is not None else [])
    n_refusals: int = sum(isinstance(s, Refusal) for s in all_responses)
    n_failures: int = sum(isinstance(s, Failure) for s in all_responses)
    malicious_scores = [
        _response_score(s) for s in malicious if isinstance(s, ResponseScore)
    ]
    control_scores = (
        [_response_score(s) for s in control if isinstance(s, ResponseScore)]
        if control is not None
        else []
    )

    # contradicting_claims is parallel to the resamples where both responses
    # were scored, in resample order
    claims_by_resample: dict[int, list[EvaluatedContradictingClaims] | Failure] = {}
    if control is not None:
        pair_indices = [
            i
            for i, (mal, ctrl) in enumerate(zip(malicious, control, strict=True))
            if isinstance(mal, ResponseScore) and isinstance(ctrl, ResponseScore)
        ]
        claims_by_resample = dict(
            zip(pair_indices, datapoint_scores.contradicting_claims)
        )

    pairs: list[str] = []
    for i, malicious_score in enumerate(malicious):
        control_score = control[i] if control is not None else None
        claims = claims_by_resample.get(i)
        pairs.append(
            f'<div class="pair"><h4>Resample {i + 1}</h4><div class="columns">'
            + _response_card("malicious", malicious_score)
            + _response_card("control", control_score)
            + "</div>"
            + (_claims_block(claims) if claims is not None else "")
            + "</div>"
        )

    summary = (
        f"<strong>#{index + 1}</strong> "
        f'<span class="snippet">{_esc(_snippet(datapoint_scores.datapoint.original_prompt))}</span><br>'
        f'<span class="summary-stats">mean score {_fmt_mean(malicious_scores)} malicious '
        f"/ {_fmt_mean(control_scores)} control · "
        f"{n_refusals} exclusion{'' if n_refusals == 1 else 's'} · "
        f"{n_failures} failure{'' if n_failures == 1 else 's'}</span>"
    )
    prompts = (
        _prompt_block("Original prompt", datapoint_scores.datapoint.original_prompt)
        + _prompt_block("Malicious prompt", datapoint_scores.datapoint.malicious_prompt)
        + _prompt_block("Control prompt", datapoint_scores.datapoint.control_prompt)
        + _rubrics_block(datapoint_scores.datapoint.rubrics)
    )
    return {
        "html": (
            f'<details class="datapoint"><summary>{summary}</summary>'
            f"{prompts}{''.join(pairs)}</details>"
        ),
        "refusals": n_refusals,
        "failures": n_failures,
    }


def _base_layout(x_title: str, y_title: str, barmode: str) -> dict:
    return {
        "barmode": barmode,
        "height": 300,
        "autosize": True,
        "margin": {"l": 60, "r": 10, "t": 30, "b": 45},
        "plot_bgcolor": "#fcfcfb",
        "paper_bgcolor": "#fcfcfb",
        "font": _FONT,
        "legend": {
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1,
            "yanchor": "bottom",
        },
        "xaxis": {
            "title": {"text": x_title},
            "showgrid": False,
            "linecolor": "#c3c2b7",
        },
        "yaxis": {
            "title": {"text": y_title},
            "gridcolor": "#e1e0d9",
            "zerolinecolor": "#c3c2b7",
            "rangemode": "tozero",
        },
    }


def _fig_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def _pair_histogram(
    control_values: list[float],
    malicious_values: list[float],
    x_title: str,
    bin_size: float,
) -> dict:
    fig = go.Figure()
    for values, name, color in [
        (control_values, "control", _CONTROL_COLOR),
        (malicious_values, "malicious", _MALICIOUS_COLOR),
    ]:
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name,
                marker_color=color,
                opacity=0.6,
                histnorm="probability",
                xbins={"size": bin_size},
            )
        )
    fig.update_layout(**_base_layout(x_title, "fraction of responses", "overlay"))
    return _fig_json(fig)


def _outcome_counts(
    responses: list[ResponseScore | Refusal | Failure],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for response in responses:
        if isinstance(response, ResponseScore):
            key = "scored"
        elif isinstance(response, Refusal):
            key = "excluded"
        else:
            key = _failure_label(response)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _outcome_figure(
    control_responses: list[ResponseScore | Refusal | Failure],
    malicious_responses: list[ResponseScore | Refusal | Failure],
) -> dict:
    control_counts = _outcome_counts(control_responses)
    malicious_counts = _outcome_counts(malicious_responses)
    categories: list[str] = ["scored", "excluded"] + sorted(
        (set(control_counts) | set(malicious_counts)) - {"scored", "excluded"}
    )
    fig = go.Figure()
    for counts, name, color in [
        (control_counts, "control", _CONTROL_COLOR),
        (malicious_counts, "malicious", _MALICIOUS_COLOR),
    ]:
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[counts.get(category, 0) for category in categories],
                name=name,
                marker_color=color,
            )
        )
    fig.update_layout(**_base_layout("outcome", "responses", "group"))
    return _fig_json(fig)


def _diff_histogram(diffs: list[float]) -> dict:
    fig = go.Figure(
        go.Histogram(
            x=diffs,
            name="malicious − control",
            marker_color=_DIFF_COLOR,
            xbins={"size": 0.1},
        )
    )
    fig.update_layout(
        **_base_layout(
            "score difference (malicious − control)", "response pairs", "overlay"
        ),
        showlegend=False,
    )
    return _fig_json(fig)


def _tile(value: str, label: str) -> str:
    return (
        f'<div class="tile"><div class="value">{_esc(value)}</div>'
        f'<div class="label">{_esc(label)}</div></div>'
    )


_CHART_DEFS: list[tuple[str, str]] = [
    ("scores", "Score distribution (scored responses)"),
    ("outcomes", "Response outcomes"),
    ("diff", "Paired score difference (both responses scored)"),
]


def _panel_html(index: int, model: Model, label: str, tiles: str) -> str:
    charts = "".join(
        f'<div class="chart-box"><h3>{_esc(chart_title)}</h3>'
        f'<div class="chart" data-fig="{chart_id}"></div></div>'
        for chart_id, chart_title in _CHART_DEFS
    )
    heading: str = label if label == model.model else f"{label} — {model.model}"
    return (
        f'<section class="model-panel" id="panel-{index}" hidden>'
        f"<h2>{_esc(heading)}</h2>"
        f'<div class="tiles">{tiles}</div>'
        f'<div class="charts">{charts}</div>'
        "<h3>Datapoints</h3>"
        '<div class="browser-controls">'
        '<label>Show <select class="filter-select">'
        '<option value="all">all datapoints</option>'
        '<option value="refusal">with ≥ 1 exclusion</option>'
        '<option value="failure">with ≥ 1 failure</option>'
        "</select></label>"
        '<button class="page-button prev-page">← prev</button>'
        '<span class="page-info"></span>'
        '<button class="page-button next-page">next →</button>'
        "</div>"
        '<div class="datapoint-list"></div>'
        "</section>"
    )


def visualize(
    models: list[Model],
    results: list[ExperimentResult],
    title: str,
    healthbench_grader: str,
    refusal_judge: str,
    statement_correctness_judge: str,
    html_filename: str,
    model_name_map: dict[str, str] | None = None,
) -> None:
    assert len(models) == len(results)

    labels: list[str] = _labels(models)
    if model_name_map is not None:
        labels = [
            model_name_map.get(model.model, model_name_map.get(label, label))
            for model, label in zip(models, labels, strict=True)
        ]

    model_payloads: list[dict] = []
    panels: list[str] = []
    for index, (model, result) in enumerate(zip(models, results, strict=True)):
        malicious_responses = [
            score
            for datapoint_scores in result.scores
            for score in datapoint_scores.malicious_scores
        ]
        control_responses = [
            score
            for datapoint_scores in result.scores
            if datapoint_scores.control_scores is not None
            for score in datapoint_scores.control_scores
        ]
        malicious_scores = [
            _response_score(s)
            for s in malicious_responses
            if isinstance(s, ResponseScore)
        ]
        control_scores = [
            _response_score(s)
            for s in control_responses
            if isinstance(s, ResponseScore)
        ]
        diffs: list[float] = [
            _response_score(mal) - _response_score(ctrl)
            for datapoint_scores in result.scores
            if datapoint_scores.control_scores is not None
            for mal, ctrl in zip(
                datapoint_scores.malicious_scores,
                datapoint_scores.control_scores,
                strict=True,
            )
            if isinstance(mal, ResponseScore) and isinstance(ctrl, ResponseScore)
        ]

        tiles = "".join(
            [
                _tile(str(len(result.scores)), "datapoints"),
                _tile(
                    f"{len(malicious_responses)} / {len(control_responses)}",
                    "responses (malicious / control)",
                ),
                _tile(
                    f"{sum(isinstance(s, Refusal) for s in malicious_responses)}"
                    f" / {sum(isinstance(s, Refusal) for s in control_responses)}",
                    "exclusions (malicious / control)",
                ),
                _tile(
                    f"{sum(isinstance(s, Failure) for s in malicious_responses)}"
                    f" / {sum(isinstance(s, Failure) for s in control_responses)}",
                    "failures (malicious / control)",
                ),
                _tile(
                    f"{_fmt_mean(malicious_scores)} / {_fmt_mean(control_scores)}",
                    "mean score (malicious / control)",
                ),
            ]
        )
        panels.append(_panel_html(index, model, labels[index], tiles))
        model_payloads.append(
            {
                "figures": {
                    "scores": _pair_histogram(
                        control_scores, malicious_scores, "HealthBench score", 0.05
                    ),
                    "outcomes": _outcome_figure(control_responses, malicious_responses),
                    "diff": _diff_histogram(diffs),
                },
                "datapoints": [
                    _datapoint_html(i, datapoint_scores)
                    for i, datapoint_scores in enumerate(result.scores)
                ],
            }
        )

    n_datapoints: int = max((len(result.scores) for result in results), default=0)
    caption = (
        f"{len(models)} model{'' if len(models) == 1 else 's'} · "
        f"{n_datapoints} datapoints · malicious and control paraphrase completions. "
        "Control responses are only generated when at least one malicious response "
        "of the datapoint is usable."
    )
    judges_caption = (
        f"HealthBench grader: {healthbench_grader} · "
        f"Refusal judge: {refusal_judge} · "
        f"Statement correctness judge: {statement_correctness_judge}"
    )
    tabs = "".join(
        f'<button class="tab-button">{_esc(label)}</button>' for label in labels
    )
    data_json = json.dumps({"models": model_payloads}, ensure_ascii=False).replace(
        "</", "<\\/"
    )

    # plot titles use "<br>" as a line separator; keep that in the heading
    title_lines: list[str] = title.split("<br>")
    title_html: str = "<br>".join(_esc(line) for line in title_lines)

    page = "".join(
        [
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_esc(' — '.join(title_lines))}</title>",
            "<script>",
            get_plotlyjs(),
            "</script><style>",
            _CSS,
            "</style></head><body><main>",
            f"<h1>{title_html}</h1>",
            f'<p class="caption">{_esc(caption)}</p>',
            f'<p class="caption">{_esc(judges_caption)}</p>',
            '<nav class="tabs">',
            tabs,
            "</nav>",
            "".join(panels),
            "</main>",
            '<script id="viz-data" type="application/json">',
            data_json,
            "</script><script>",
            _JS,
            "</script></body></html>",
        ]
    )

    with open(html_filename, "w") as f:
        f.write(page)
