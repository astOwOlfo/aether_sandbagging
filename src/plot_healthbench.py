"""Compare healthbench sandbagging results across models in a single figure.

Usage:
    from src.plot_healthbench import plot
    fig = plot(
        models, results, title, healthbench_grader, refusal_judge,
        statement_correctness_judge,
    )
    fig.write_html("visualizations/comparison.html")

Three subplots in a row: grouped control/malicious bar pairs per model for
average rubric score, refusal fraction, and fraction of contradicting claims
that are correct. All bars carry 95% error bars and all y axes span [0, 1].

Unlike ScoreSummary (which averages per datapoint first), the score subplot
averages unweighted over all (datapoint, resample) pairs where both the
malicious and the control score of that resample are valid.

Error bars use cluster-robust standard errors with datapoints as clusters:
resamples (and extracted claims) of the same datapoint are correlated, so the
effective sample size is governed by the number of datapoints, not the number
of observations.

P values annotated above the score bars come from one-sided cluster-robust z
tests with the same clustering of whether the per-resample paired score
difference (malicious minus control) is negative.
"""

from collections.abc import Callable
from math import isnan, nan, sqrt
from statistics import NormalDist

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.healthbench import (
    DatapointScores,
    ExperimentResult,
    Failure,
    Refusal,
    ResponseScore,
)
from src.llm_apis import Model

_Z95 = NormalDist().inv_cdf(0.975)

_CONTROL_COLOR = "#2a78d6"
_MALICIOUS_COLOR = "#eb6834"


def _response_score(score: ResponseScore) -> float:
    return sum(s.rubric.points for s in score.rubric_scores if s.met) / sum(
        s.rubric.points for s in score.rubric_scores if s.rubric.points > 0
    )


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
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _add_p_annotation(
    fig: go.Figure, label: str, bar_top: float, p: float, row: int, col: int
) -> None:
    """Write a p value just above the error bar of the bar (or bar group) at
    `label`, skipping undefined p values."""
    if isnan(p) or isnan(bar_top):
        return
    fig.add_annotation(
        text=_format_p(p),
        x=label,
        y=min(bar_top + 0.03, 1.0),
        yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": "#52514e"},
        row=row,
        col=col,
    )


def _score_clusters(
    scores: list[DatapointScores], malicious: bool
) -> list[list[float]]:
    """One cluster per datapoint, containing the malicious (or control) score of
    every resample where both the malicious and the control score are valid."""
    clusters: list[list[float]] = []
    for datapoint_scores in scores:
        if datapoint_scores.control_scores is None:
            continue
        clusters.append(
            [
                _response_score(malicious_score if malicious else control_score)
                for malicious_score, control_score in zip(
                    datapoint_scores.malicious_scores,
                    datapoint_scores.control_scores,
                    strict=True,
                )
                if isinstance(malicious_score, ResponseScore)
                and isinstance(control_score, ResponseScore)
            ]
        )
    return clusters


def _score_diff_clusters(scores: list[DatapointScores]) -> list[list[float]]:
    """One cluster per datapoint, containing the malicious minus control score
    difference of every resample where both scores are valid."""
    clusters: list[list[float]] = []
    for datapoint_scores in scores:
        if datapoint_scores.control_scores is None:
            continue
        clusters.append(
            [
                _response_score(malicious_score) - _response_score(control_score)
                for malicious_score, control_score in zip(
                    datapoint_scores.malicious_scores,
                    datapoint_scores.control_scores,
                    strict=True,
                )
                if isinstance(malicious_score, ResponseScore)
                and isinstance(control_score, ResponseScore)
            ]
        )
    return clusters


def _refusal_clusters(
    scores: list[DatapointScores], malicious: bool
) -> list[list[float]]:
    """One cluster per datapoint, with a 0/1 refusal indicator per completion."""
    clusters: list[list[float]] = []
    for datapoint_scores in scores:
        completion_scores = (
            datapoint_scores.malicious_scores
            if malicious
            else datapoint_scores.control_scores
        )
        if completion_scores is None:
            continue
        clusters.append(
            [float(isinstance(score, Refusal)) for score in completion_scores]
        )
    return clusters


def _claim_correct_clusters(
    scores: list[DatapointScores], malicious: bool
) -> list[list[float]]:
    """One cluster per datapoint, with a 0/1 correctness indicator per
    contradicting claim pair (pooled over all resample pairs)."""
    clusters: list[list[float]] = []
    for datapoint_scores in scores:
        clusters.append(
            [
                float(claims.malicious_correct if malicious else claims.control_correct)
                for claims_list in datapoint_scores.contradicting_claims
                if not isinstance(claims_list, Failure)
                for claims in claims_list
            ]
        )
    return clusters


def plot(
    models: list[Model],
    results: list[ExperimentResult],
    title: str,
    healthbench_grader: str,
    refusal_judge: str,
    statement_correctness_judge: str,
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

    # Each subplot: title, per-condition cluster function, and — for the one
    # that gets a p value — the paired malicious-minus-control difference
    # cluster function with the alternative hypothesis of its z test against 0.
    subplots: list[
        tuple[
            str,
            Callable[[list[DatapointScores], bool], list[list[float]]],
            Callable[[list[DatapointScores]], list[list[float]]] | None,
            str | None,
        ]
    ] = [
        ("Average HealthBench score", _score_clusters, _score_diff_clusters, "less"),
        ("Fraction excluded", _refusal_clusters, None, None),
        (
            "Fraction of statements on which<br>responses contradict each other<br>that are correct",
            _claim_correct_clusters,
            None,
            None,
        ),
    ]

    positions: list[tuple[int, int]] = [(1, 1), (1, 2), (1, 3)]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[subplot_title for subplot_title, *_ in subplots],
        horizontal_spacing=0.08,
    )

    for (row, col), (_, cluster_fn, diff_cluster_fn, alternative) in zip(
        positions, subplots, strict=True
    ):
        # highest error bar tip of each control/malicious bar pair, where the
        # pair's p value annotation goes
        group_tops: list[float] = [0.0] * len(results)
        for malicious, name, color in [
            (False, "control paraphrases", _CONTROL_COLOR),
            (True, "malicious paraphrases", _MALICIOUS_COLOR),
        ]:
            means = []
            ci95s = []
            for i, result in enumerate(results):
                mean, se = _cluster_mean_and_se(cluster_fn(result.scores, malicious))
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
                    showlegend=(row, col) == positions[0],
                ),
                row=row,
                col=col,
            )
        if diff_cluster_fn is None or alternative is None:
            continue
        for label, group_top, result in zip(labels, group_tops, results, strict=True):
            diff_mean, diff_se = _cluster_mean_and_se(diff_cluster_fn(result.scores))
            _add_p_annotation(
                fig,
                label,
                group_top,
                _z_test_p(diff_mean, 0.0, diff_se, alternative),
                row=row,
                col=col,
            )

    # The top margin stacks, from the top: the title, the legend, and the
    # grader/judge caption, above the (three line tall) row 1 subplot titles.
    # Everything is laid out in pixels from the top of the figure and converted
    # to the coordinates plotly wants.
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
    _SUBPLOT_TITLES_HEIGHT = 48  # three lines
    _PLOT_AREA_HEIGHT = 380
    _MARGIN_BOTTOM = 90

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
        # a legend title above (or left of) the entries, so the grader/judge
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
        text=f"HealthBench grader: {healthbench_grader}<br>"
        f"Refusal judge: {refusal_judge}<br>"
        f"Statement correctness judge: {statement_correctness_judge}",
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

    if html_filename is not None:
        fig.write_html(html_filename, include_plotlyjs=True)

    return fig
