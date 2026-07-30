"""Browsable HTML report of every completion in a literature review experiment.

Usage:
    from src.visualize_literature_review import visualize
    visualize(
        models, results, title, claim_extractor, correctness_judge,
        refusal_judge, "visualizations/literature_review.html",
    )

One tab per model. Each tab shows summary stat tiles, distributions (response
outcomes, papers and claims per response, fraction of papers that exist,
fraction of claims supported by their paper), and a filterable, paginated
browser over every datapoint with the full malicious and control completions
(including refusals), reasoning, and the per-response evaluation summary.

`ExperimentResult` does not store the prompts, so `visualize` reloads the
dataset with `load_literature_review_data()` to show them; pass `dataset=` if
the experiment ran on anything but the default dataset. The order of
`result.evaluated_datapoints` matches the dataset order.

The file is self-contained: plotly.js is embedded, and the datapoint browser is
stored as JSON and rendered client-side a page at a time so the page stays
responsive with thousands of completions.
"""

import json
from html import escape
from statistics import fmean

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from src.literature_review import (
    Datapoint,
    EvaluatedDatapoint,
    EvaluatedResponse,
    ExperimentResult,
    Failure,
    Refusal,
    ResponseSummary,
    load_literature_review_data,
)
from src.llm_apis import Completion, Model

_CONTROL_COLOR = "#2a78d6"
_MALICIOUS_COLOR = "#eb6834"

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
table.summary { border-collapse: collapse; font-size: 13px; margin: 6px 0; }
table.summary td { border: 1px solid #e1e0d9; padding: 3px 8px; }
table.summary td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.indented { padding-left: 18px; }
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


def _fmt_mean(values: list[float]) -> str:
    if not values:
        return "–"
    return f"{fmean(values):.2f}"


def _failure_label(failure: Failure) -> str:
    return failure.name.replace("_", " ").lower()


def _snippet(prompt: str, limit: int = 120) -> str:
    text = " ".join(prompt.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _badge(kind: str, text: str) -> str:
    return f'<span class="badge badge-{kind}">{_esc(text)}</span>'


def _prompt_block(title: str, prompt: str) -> str:
    return (
        f'<details class="sub"><summary>{_esc(title)}</summary>'
        f"<pre>{_esc(prompt)}</pre></details>"
    )


def _completion_block(response: Completion) -> str:
    parts: list[str] = []
    if response.reasoning is not None:
        parts.append(
            '<details class="sub"><summary>Reasoning</summary>'
            f"<pre>{_esc(response.reasoning)}</pre></details>"
        )
    parts.append(f"<pre>{_esc(response.completion)}</pre>")
    return "".join(parts)


def _fraction_papers_not_hallucinated(summary: ResponseSummary) -> float | None:
    """Fraction of the papers the existence judge ruled on that exist; None if
    the judge ruled on no paper."""
    n_judged: int = summary.n_papers - summary.n_paper_judge_failed
    if n_judged == 0:
        return None
    return summary.n_correct_papers / n_judged


def _fraction_claims_supported(summary: ResponseSummary) -> float | None:
    """Fraction of the claims the correctness judge ruled on that are supported
    by the paper they are attributed to; None if the judge ruled on no claim."""
    n_judged: int = summary.n_existing_paper_claims - summary.n_claim_judge_failed
    if n_judged == 0:
        return None
    return summary.n_supported_paper_claims / n_judged


def _summary_table(summary: ResponseSummary) -> str:
    rows: list[tuple[str, int, bool]] = [
        ("papers cited", summary.n_papers, False),
        ("exist", summary.n_correct_papers, True),
        ("hallucinated", summary.n_hallucinated_papers, True),
        ("existence judge failed", summary.n_paper_judge_failed, True),
        ("claims", summary.n_claims, False),
        ("attributed to a paper", summary.n_paper_claims, True),
        ("paperless", summary.n_paperless_claims, True),
        ("claims on existing papers", summary.n_existing_paper_claims, False),
        ("supported by the paper", summary.n_supported_paper_claims, True),
        ("unsupported", summary.n_unsupported_paper_claims, True),
        ("support judge failed", summary.n_claim_judge_failed, True),
    ]
    body = "".join(
        f"<tr><td{' class=indented' if indented else ''}>{_esc(name)}</td>"
        f"<td>{count}</td></tr>"
        for name, count, indented in rows
    )
    return (
        '<details class="sub" open><summary>Evaluation summary</summary>'
        f'<table class="summary">{body}</table></details>'
    )


def _response_card(
    side: str, response: EvaluatedResponse | Refusal | Failure | None
) -> str:
    if response is None:
        badge = _badge("missing", "not generated")
        body = (
            '<p class="note">The control response is not generated when the '
            "malicious response of the pair is an exclusion or a failure.</p>"
        )
    elif isinstance(response, EvaluatedResponse):
        badge = _badge("scored", "evaluated")
        body = _completion_block(response.response) + _summary_table(response.summary)
    elif isinstance(response, Refusal):
        badge = _badge("refused", "excluded")
        if response.response is None:
            body = '<p class="note">Excluded via provider content filter — no completion was returned.</p>'
        else:
            body = _completion_block(response.response)
    else:
        badge = _badge("failure", f"failure: {_failure_label(response)}")
        body = '<p class="note">No usable evaluation for this response.</p>'
    return f'<div class="response {side}"><h5>{side}{badge}</h5>{body}</div>'


def _datapoint_html(
    index: int, datapoint: Datapoint, evaluated: EvaluatedDatapoint
) -> dict:
    pairs = evaluated.evaluated_response_pairs
    all_responses = [pair.malicious for pair in pairs] + [
        pair.control for pair in pairs if pair.control is not None
    ]
    n_refusals: int = sum(isinstance(r, Refusal) for r in all_responses)
    n_failures: int = sum(isinstance(r, Failure) for r in all_responses)
    n_evaluated: int = sum(isinstance(r, EvaluatedResponse) for r in all_responses)

    pair_blocks: list[str] = []
    for i, pair in enumerate(pairs):
        pair_blocks.append(
            f'<div class="pair"><h4>Resample {i + 1}</h4><div class="columns">'
            + _response_card("malicious", pair.malicious)
            + _response_card("control", pair.control)
            + "</div></div>"
        )

    summary = (
        f"<strong>#{index + 1}</strong> "
        f'<span class="snippet">{_esc(_snippet(datapoint.malicious_prompt))}</span><br>'
        f'<span class="summary-stats">{n_evaluated} evaluated · '
        f"{n_refusals} exclusion{'' if n_refusals == 1 else 's'} · "
        f"{n_failures} failure{'' if n_failures == 1 else 's'}</span>"
    )
    prompts = _prompt_block(
        "Malicious prompt", datapoint.malicious_prompt
    ) + _prompt_block("Control prompt", datapoint.control_prompt)
    return {
        "html": (
            f'<details class="datapoint"><summary>{summary}</summary>'
            f"{prompts}{''.join(pair_blocks)}</details>"
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
    responses: list[EvaluatedResponse | Refusal | Failure],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for response in responses:
        if isinstance(response, EvaluatedResponse):
            key = "evaluated"
        elif isinstance(response, Refusal):
            key = "excluded"
        else:
            key = _failure_label(response)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _outcome_figure(
    control_responses: list[EvaluatedResponse | Refusal | Failure],
    malicious_responses: list[EvaluatedResponse | Refusal | Failure],
) -> dict:
    control_counts = _outcome_counts(control_responses)
    malicious_counts = _outcome_counts(malicious_responses)
    categories: list[str] = ["evaluated", "excluded"] + sorted(
        (set(control_counts) | set(malicious_counts)) - {"evaluated", "excluded"}
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


def _tile(value: str, label: str) -> str:
    return (
        f'<div class="tile"><div class="value">{_esc(value)}</div>'
        f'<div class="label">{_esc(label)}</div></div>'
    )


_CHART_DEFS: list[tuple[str, str]] = [
    ("outcomes", "Response outcomes"),
    ("papers", "Papers per response (evaluated responses)"),
    ("claims", "Claims per response (evaluated responses)"),
    ("papers_exist", "Fraction of judged papers that exist, per response"),
    (
        "claims_supported",
        "Fraction of judged claims supported by their paper, per response",
    ),
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
    claim_extractor: str,
    correctness_judge: str,
    refusal_judge: str,
    html_filename: str,
    model_name_map: dict[str, str] | None = None,
    dataset: list[Datapoint] | None = None,
) -> None:
    assert len(models) == len(results)

    if dataset is None:
        dataset = load_literature_review_data()
    for result in results:
        assert len(result.evaluated_datapoints) == len(dataset), (
            "results and dataset lengths differ; pass the dataset the experiment "
            "actually ran on via the `dataset` argument"
        )

    labels: list[str] = _labels(models)
    if model_name_map is not None:
        labels = [
            model_name_map.get(model.model, model_name_map.get(label, label))
            for model, label in zip(models, labels, strict=True)
        ]

    model_payloads: list[dict] = []
    panels: list[str] = []
    for index, (model, result) in enumerate(zip(models, results, strict=True)):
        pairs = [
            pair
            for evaluated in result.evaluated_datapoints
            for pair in evaluated.evaluated_response_pairs
        ]
        malicious_responses = [pair.malicious for pair in pairs]
        control_responses = [pair.control for pair in pairs if pair.control is not None]
        malicious_summaries = [
            r.summary for r in malicious_responses if isinstance(r, EvaluatedResponse)
        ]
        control_summaries = [
            r.summary for r in control_responses if isinstance(r, EvaluatedResponse)
        ]

        def _values(summaries: list[ResponseSummary], value_fn) -> list[float]:
            return [
                value
                for summary in summaries
                if (value := value_fn(summary)) is not None
            ]

        malicious_papers = _values(malicious_summaries, lambda s: float(s.n_papers))
        control_papers = _values(control_summaries, lambda s: float(s.n_papers))
        malicious_claims = _values(malicious_summaries, lambda s: float(s.n_claims))
        control_claims = _values(control_summaries, lambda s: float(s.n_claims))
        malicious_exist = _values(
            malicious_summaries, _fraction_papers_not_hallucinated
        )
        control_exist = _values(control_summaries, _fraction_papers_not_hallucinated)
        malicious_supported = _values(malicious_summaries, _fraction_claims_supported)
        control_supported = _values(control_summaries, _fraction_claims_supported)

        tiles = "".join(
            [
                _tile(str(len(result.evaluated_datapoints)), "datapoints"),
                _tile(
                    f"{len(malicious_responses)} / {len(control_responses)}",
                    "responses (malicious / control)",
                ),
                _tile(
                    f"{sum(isinstance(r, Refusal) for r in malicious_responses)}"
                    f" / {sum(isinstance(r, Refusal) for r in control_responses)}",
                    "exclusions (malicious / control)",
                ),
                _tile(
                    f"{sum(isinstance(r, Failure) for r in malicious_responses)}"
                    f" / {sum(isinstance(r, Failure) for r in control_responses)}",
                    "failures (malicious / control)",
                ),
                _tile(
                    f"{_fmt_mean(malicious_papers)} / {_fmt_mean(control_papers)}",
                    "mean papers per response (malicious / control)",
                ),
                _tile(
                    f"{_fmt_mean(malicious_claims)} / {_fmt_mean(control_claims)}",
                    "mean claims per response (malicious / control)",
                ),
                _tile(
                    f"{_fmt_mean(malicious_supported)} / {_fmt_mean(control_supported)}",
                    "mean fraction of claims supported (malicious / control)",
                ),
            ]
        )
        panels.append(_panel_html(index, model, labels[index], tiles))
        model_payloads.append(
            {
                "figures": {
                    "outcomes": _outcome_figure(control_responses, malicious_responses),
                    "papers": _pair_histogram(
                        control_papers, malicious_papers, "papers per response", 1
                    ),
                    "claims": _pair_histogram(
                        control_claims, malicious_claims, "claims per response", 2
                    ),
                    "papers_exist": _pair_histogram(
                        control_exist,
                        malicious_exist,
                        "fraction of papers that exist",
                        0.1,
                    ),
                    "claims_supported": _pair_histogram(
                        control_supported,
                        malicious_supported,
                        "fraction of claims supported",
                        0.1,
                    ),
                },
                "datapoints": [
                    _datapoint_html(i, datapoint, evaluated)
                    for i, (datapoint, evaluated) in enumerate(
                        zip(dataset, result.evaluated_datapoints, strict=True)
                    )
                ],
            }
        )

    n_datapoints: int = len(dataset)
    caption = (
        f"{len(models)} model{'' if len(models) == 1 else 's'} · "
        f"{n_datapoints} datapoints · malicious and control paraphrase completions. "
        "The control response of a pair is only generated when the malicious "
        "response is neither an exclusion nor a failure."
    )
    judges_caption = (
        f"Claim extractor: {claim_extractor} · "
        f"Correctness judge: {correctness_judge} · "
        f"Refusal judge: {refusal_judge}"
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
