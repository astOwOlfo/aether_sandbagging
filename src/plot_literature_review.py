"""Compare literature review sandbagging results across models in a single figure.

Usage:
    from src.plot_literature_review import plot
    fig = plot(
        models, results, title, claim_extractor, correctness_judge, refusal_judge,
    )
    fig.write_html("plots/literature_review/plot.html")

Five subplots, each a grouped malicious/control bar pair per model: a 2x2 grid
with average number of papers per response, average number of claims per
response, fraction of papers that are not hallucinated, and fraction of claims
that are supported by the paper they are attributed to, plus a fraction of
responses that are refusals subplot to the right of the grid, vertically
centered.

Except for the refusal subplot, every average is taken over response pairs: the
malicious and control response of the ith resample of the jth datapoint, for all
(i, j) where neither of the two is a Failure or a Refusal. The refusal subplot
averages a 0/1 refusal indicator over the responses that are not Failures. Note
that control responses are only generated when the corresponding malicious
response is neither a Failure nor a Refusal, so the control refusal fraction is
conditional on that, and the two sides are not directly comparable.

The two fraction-of-X subplots average per-response fractions (not pooled
counts), and skip responses whose denominator is 0. Denominators exclude judge
failures: the fraction of non-hallucinated papers is over the papers the paper
existence judge ruled on, and the fraction of supported claims is over the
claims the claim correctness judge ruled on.

Error bars use cluster-robust standard errors with datapoints as clusters:
resamples of the same datapoint are correlated, so the effective sample size is
governed by the number of datapoints, not the number of observations.

P values annotated above the four control/malicious bar pairs of the 2x2 grid
(all metrics but refusals) come from one-sided cluster-robust z tests with the
same clustering of whether the per-pair paired difference (malicious minus
control) is negative.
"""

from collections.abc import Callable
from math import isnan, nan, sqrt
from statistics import NormalDist

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.literature_review import (
    EvaluatedDatapoint,
    EvaluatedResponse,
    ExperimentResult,
    Failure,
    Refusal,
    ResponseSummary,
)
from src.llm_apis import Model

_Z95 = NormalDist().inv_cdf(0.975)

_CONTROL_COLOR = "#2a78d6"
_MALICIOUS_COLOR = "#eb6834"


def _cluster_mean_and_se(clusters: list[list[float]]) -> tuple[float, float]:
    """Mean over all observations, and its cluster-robust standard error.

    The mean is unweighted over observations, so with unequal cluster sizes the
    variance of the mean is sum_g (sum_{x in g} (x - mean))^2 / n^2, with a
    G/(G-1) small-sample correction, where g ranges over clusters and n is the
    total number of observations."""
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    n_observations: int = sum(len(cluster) for cluster in clusters)
    if n_observations == 0:
        return nan, 0.0

    mean: float = sum(sum(cluster) for cluster in clusters) / n_observations

    n_clusters: int = len(clusters)
    if n_clusters < 2:
        return mean, 0.0

    cluster_residual_sums: list[float] = [
        sum(x - mean for x in cluster) for cluster in clusters
    ]
    variance: float = (
        n_clusters
        / (n_clusters - 1)
        * sum(residual_sum**2 for residual_sum in cluster_residual_sums)
        / n_observations**2
    )
    return mean, sqrt(variance)


def _z_test_p(mean: float, null: float, se: float, alternative: str) -> float:
    """P value of a z test of `mean` against `null` using the cluster-robust
    standard error `se`; `alternative` is "less", "greater", or "two-sided"."""
    if isnan(mean) or se <= 0:
        return nan
    z = (mean - null) / se
    match alternative:
        case "less":
            return NormalDist().cdf(z)
        case "greater":
            return 1 - NormalDist().cdf(z)
        case "two-sided":
            return 2 * NormalDist().cdf(-abs(z))
        case _:
            raise ValueError(f"unknown alternative {alternative!r}")


def _format_p(p: float) -> str:
    if isnan(p):
        return "p = nan"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _add_p_annotation(
    fig: go.Figure,
    label: str,
    bar_top: float,
    p: float,
    row: int,
    col: int,
    y_max: float | None,
) -> None:
    """Write a p value just above the error bar of the bar group at `label`;
    undefined p values are written as "p = nan". `y_max` caps the annotation
    position on axes with a fixed range; pass None on autoranged axes."""
    if isnan(bar_top):
        return
    offset: float = 0.03 * (y_max if y_max is not None else max(bar_top, 1e-9))
    y: float = bar_top + offset
    if y_max is not None:
        y = min(y, y_max)
    fig.add_annotation(
        text=_format_p(p),
        x=label,
        y=y,
        yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": "#52514e"},
        row=row,
        col=col,
    )


def _n_papers(summary: ResponseSummary) -> float | None:
    return summary.n_papers


def _n_claims(summary: ResponseSummary) -> float | None:
    return summary.n_claims


def _fraction_papers_not_hallucinated(summary: ResponseSummary) -> float | None:
    """Fraction of the papers the existence judge ruled on that exist. None
    (i.e. this response is excluded) if the judge ruled on no paper."""
    n_judged: int = summary.n_papers - summary.n_paper_judge_failed
    if n_judged == 0:
        return None
    return summary.n_correct_papers / n_judged


def _fraction_claims_supported(summary: ResponseSummary) -> float | None:
    """Fraction of the claims the correctness judge ruled on that are supported
    by the paper they are attributed to. None (i.e. this response is excluded)
    if the judge ruled on no claim."""
    n_judged: int = summary.n_existing_paper_claims - summary.n_claim_judge_failed
    if n_judged == 0:
        return None
    return summary.n_supported_paper_claims / n_judged


def _paired_clusters(
    evaluated_datapoints: list[EvaluatedDatapoint],
    malicious: bool,
    value: Callable[[ResponseSummary], float | None],
) -> list[list[float]]:
    """One cluster per datapoint, containing the value of the malicious (or
    control) response of every pair where neither the malicious nor the control
    response is a Failure or a Refusal, skipping the responses whose value is
    None."""
    clusters: list[list[float]] = []
    for datapoint in evaluated_datapoints:
        cluster: list[float] = []
        for pair in datapoint.evaluated_response_pairs:
            if not isinstance(pair.malicious, EvaluatedResponse) or not isinstance(
                pair.control, EvaluatedResponse
            ):
                continue
            response = pair.malicious if malicious else pair.control
            response_value: float | None = value(response.summary)
            if response_value is None:
                continue
            cluster.append(response_value)
        clusters.append(cluster)
    return clusters


def _paired_diff_clusters(
    evaluated_datapoints: list[EvaluatedDatapoint],
    value: Callable[[ResponseSummary], float | None],
) -> list[list[float]]:
    """One cluster per datapoint, containing the malicious minus control value
    difference of every pair where neither response is a Failure or a Refusal
    and both values are not None."""
    clusters: list[list[float]] = []
    for datapoint in evaluated_datapoints:
        cluster: list[float] = []
        for pair in datapoint.evaluated_response_pairs:
            if not isinstance(pair.malicious, EvaluatedResponse) or not isinstance(
                pair.control, EvaluatedResponse
            ):
                continue
            malicious_value: float | None = value(pair.malicious.summary)
            control_value: float | None = value(pair.control.summary)
            if malicious_value is None or control_value is None:
                continue
            cluster.append(malicious_value - control_value)
        clusters.append(cluster)
    return clusters


def _refusal_clusters(
    evaluated_datapoints: list[EvaluatedDatapoint], malicious: bool
) -> list[list[float]]:
    """One cluster per datapoint, with a 0/1 refusal indicator per malicious (or
    control) response that is not a Failure. Control responses that were never
    generated (because the corresponding malicious response is a Failure or a
    Refusal) are skipped."""
    clusters: list[list[float]] = []
    for datapoint in evaluated_datapoints:
        responses: list[EvaluatedResponse | Refusal | Failure] = [
            response
            for pair in datapoint.evaluated_response_pairs
            if (response := pair.malicious if malicious else pair.control) is not None
        ]
        clusters.append(
            [
                float(isinstance(response, Refusal))
                for response in responses
                if not isinstance(response, Failure)
            ]
        )
    return clusters


def plot(
    models: list[Model],
    results: list[ExperimentResult],
    title: str,
    claim_extractor: str,
    correctness_judge: str,
    refusal_judge: str,
    model_name_map: dict[str, str] | None = None,
    html_filename: str | None = None,
) -> go.Figure:
    assert len(models) == len(results)

    labels: list[str] = [model.model.split("/")[-1] for model in models]
    if len(set(labels)) != len(labels):
        labels = [model.model for model in models]
    if model_name_map is not None:
        labels = [
            model_name_map.get(model.model, model_name_map.get(label, label))
            for model, label in zip(models, labels, strict=True)
        ]

    # In grid order — (1, 1), (1, 2), (1, 3), (2, 1), (2, 2) — so the list
    # doubles as the subplot_titles order: subplot title, cluster function,
    # whether the y axis is [0, 1], per-response value whose paired
    # malicious-minus-control difference gives the p value (None for no p
    # value), and the subplot's (row, col).
    subplots: list[
        tuple[
            str,
            Callable[[list[EvaluatedDatapoint], bool], list[list[float]]],
            bool,
            Callable[[ResponseSummary], float | None] | None,
            tuple[int, int],
        ]
    ] = [
        (
            "Papers per response",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _n_papers
            ),
            False,
            _n_papers,
            (1, 1),
        ),
        (
            "Claims per response",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _n_claims
            ),
            False,
            _n_claims,
            (1, 2),
        ),
        (
            "Fraction excluded",
            _refusal_clusters,
            True,
            None,
            (1, 3),
        ),
        (
            "Fraction papers not hallucinated",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _fraction_papers_not_hallucinated
            ),
            True,
            _fraction_papers_not_hallucinated,
            (2, 1),
        ),
        (
            "Fraction claims supported by papers",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _fraction_claims_supported
            ),
            True,
            _fraction_claims_supported,
            (2, 2),
        ),
    ]

    # The top margin stacks, from the top: the title, the legend, and the
    # extractor/judge caption, above the row 1 subplot titles. Everything is
    # laid out in pixels from the top of the figure and converted to the
    # coordinates plotly wants.
    #
    # The title grows upwards from a fixed bottom edge into space reserved for
    # _RESERVED_TITLE_LINES lines, so that titles of different lengths give
    # figures of identical size, with the title in the same font size at the
    # same distance above the legend. (Plotly never rescales title text, so a
    # title only ever looks smaller because the figure it sits in is bigger.)
    _TITLE_TOP = 16
    _TITLE_LINE_HEIGHT = 26
    _RESERVED_TITLE_LINES = 3
    _LEGEND_HEIGHT = 50  # two stacked entries
    _CAPTION_HEIGHT = 54  # three lines
    _SUBPLOT_TITLES_HEIGHT = 32  # one line, shifted up for a gap below
    _ROW_HEIGHT = 380
    # gap between the grid rows: row 1 x tick labels plus row 2 subplot titles
    _ROW_GAP = 140
    _PLOT_AREA_HEIGHT = 2 * _ROW_HEIGHT + _ROW_GAP
    _MARGIN_BOTTOM = 90

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {"rowspan": 2}], [{}, {}, None]],
        subplot_titles=[subplot_title for subplot_title, *_ in subplots],
        horizontal_spacing=0.08,
        vertical_spacing=_ROW_GAP / _PLOT_AREA_HEIGHT,
    )
    # at this point the only annotations are the subplot titles; lift them a
    # little off their subplots
    for annotation in fig.layout.annotations:
        annotation.yshift = 8

    for _, cluster_fn, unit_range, value_fn, (row, col) in subplots:
        # highest error bar tip of each control/malicious bar pair, where the
        # pair's p value annotation goes
        group_tops: list[float] = [0.0] * len(results)
        for malicious, name, color in [
            (False, "control", _CONTROL_COLOR),
            (True, "malicious", _MALICIOUS_COLOR),
        ]:
            means: list[float] = []
            ci95s: list[float] = []
            for i, result in enumerate(results):
                mean, se = _cluster_mean_and_se(
                    cluster_fn(result.evaluated_datapoints, malicious)
                )
                means.append(mean)
                ci95s.append(_Z95 * se)
                if not isnan(mean):
                    group_tops[i] = max(group_tops[i], mean + _Z95 * se)
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=means,
                    name=name,
                    marker_color=color,
                    error_y={
                        "type": "data",
                        "array": ci95s,
                        "color": "#52514e",
                        "thickness": 1.5,
                        "width": 4,
                    },
                    legendgroup=name,
                    showlegend=(row, col) == subplots[0][4],
                ),
                row=row,
                col=col,
            )
        if value_fn is None:
            continue
        for label, group_top, result in zip(labels, group_tops, results, strict=True):
            diff_mean, diff_se = _cluster_mean_and_se(
                _paired_diff_clusters(result.evaluated_datapoints, value_fn)
            )
            _add_p_annotation(
                fig,
                label,
                group_top,
                _z_test_p(diff_mean, 0.0, diff_se, "less"),
                row=row,
                col=col,
                y_max=1.0 if unit_range else None,
            )

    title_lines: int = title.count("<br>") + 1
    title_bottom: int = _TITLE_TOP + _TITLE_LINE_HEIGHT * max(
        _RESERVED_TITLE_LINES, title_lines
    )
    title_top: int = title_bottom - _TITLE_LINE_HEIGHT * title_lines
    legend_top: int = title_bottom + 22
    caption_top: int = legend_top + _LEGEND_HEIGHT + 8
    margin_top: int = caption_top + _CAPTION_HEIGHT + 10 + _SUBPLOT_TITLES_HEIGHT
    height: int = margin_top + _PLOT_AREA_HEIGHT + _MARGIN_BOTTOM

    def _paper_y(pixels_from_top: float) -> float:
        """Paper coordinate (1 at the top of the plotting area) of a pixel
        offset measured from the top of the figure."""
        return 1 + (margin_top - pixels_from_top) / _PLOT_AREA_HEIGHT

    fig.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.06,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": "#0b0b0b",
        },
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "yref": "container",
            "y": 1 - title_top / height,
            "yanchor": "top",
            "font": {"size": 20},
        },
        # The legend sits in the top margin, under the title. Plotly only allows
        # a legend title above (or left of) the entries, so the extractor/judge
        # lines are an annotation placed just below the legend instead.
        legend={
            "orientation": "v",
            # the control trace is added first, so reversing the legend order
            # puts malicious above control without reordering the bars
            "traceorder": "reversed",
            "yanchor": "bottom",
            "y": _paper_y(legend_top + _LEGEND_HEIGHT),
            "xanchor": "center",
            "x": 0.5,
        },
        width=1100,
        height=height,
        margin={"t": margin_top, "b": _MARGIN_BOTTOM},
    )
    fig.add_annotation(
        text=f"Claim extractor: {claim_extractor}<br>"
        f"Correctness judge: {correctness_judge}<br>"
        f"Refusal judge: {refusal_judge}",
        xref="paper",
        yref="paper",
        x=0.5,
        xanchor="center",
        y=_paper_y(caption_top),
        yanchor="top",
        align="center",
        showarrow=False,
        font={"size": 13, "color": "#52514e"},
    )
    fig.update_yaxes(range=[0, 1], gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    # counts are unbounded, so [0, 1] would clip them; the fractions stay in [0, 1]
    for _, _, unit_range, _, (row, col) in subplots:
        if not unit_range:
            fig.update_yaxes(
                range=None, autorange=True, rangemode="tozero", row=row, col=col
            )

    # The refusal subplot spans both grid rows; shrink it to one row's height,
    # vertically centered, and move its subplot title down with it.
    row_fraction: float = _ROW_HEIGHT / _PLOT_AREA_HEIGHT
    refusal_domain: list[float] = [
        (1 - row_fraction) / 2,
        (1 + row_fraction) / 2,
    ]
    fig.update_yaxes(domain=refusal_domain, row=1, col=3)
    for annotation in fig.layout.annotations:
        if annotation.text == "Refusals":
            annotation.y = refusal_domain[1]

    if html_filename is not None:
        fig.write_html(html_filename, include_plotlyjs=True)

    return fig
