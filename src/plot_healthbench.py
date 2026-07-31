"""Compare healthbench sandbagging results across models in a single figure.

Usage:
    from src.plot_healthbench import plot
    fig = plot(
        models, results, title, healthbench_grader, refusal_judge,
        statement_correctness_judge,
    )
    fig.write_html("visualizations/comparison.html")

Three subplots in a row: grouped control/malicious bar pairs per model for
average rubric score and fraction of contradicting claims that are correct, and,
to their right, a single bar per model for the fraction of completion pairs
excluded because the malicious or the control completion is a refusal (a
property of the pair, hence one bar rather than a pair of bars; pairs where
either completion is a judge or generation failure are left out of both the
numerator and the denominator, since they are excluded for a different reason).
All bars carry 95% error bars and all y axes span [0, 1].

A tint behind the plotting area says which family a subplot's metric belongs to:
how detailed the response is, or how good what it says is. The exclusion
fraction is neither, so that subplot keeps the plain background. A second legend,
beside the control/malicious one, is the key to the two tints.

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
# subplots whose metric is a property of a completion pair get a single neutral
# bar per model instead of a control/malicious pair
_PAIR_COLOR = "#8b8a82"
# width of one bar of a control/malicious pair, in category units, given the
# bargap and bargroupgap of the layout below; single bars match it so that they
# do not fill their whole slot
_SINGLE_BAR_WIDTH = 0.305

# A wash behind the plotting area tells the two families of metrics apart: how
# much the response says versus how good what it says is. The exclusion subplot
# measures neither, so it keeps the plain background. The tints are nearly
# neutral so that they do not compete with the control/malicious colours; the
# legend swatches use the same tints, outlined so that they read at swatch size.
_DETAILEDNESS_TINT = "#f3ecdc"
_DETAILEDNESS_TINT_EDGE = "#cbbc98"
_QUALITY_TINT = "#e6e9f5"
_QUALITY_TINT_EDGE = "#b3bcdf"


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


def _excluded_clusters(scores: list[DatapointScores]) -> list[list[float]]:
    """One cluster per datapoint, with a 0/1 indicator per completion pair of
    whether the pair is excluded because the malicious or the control completion
    is a refusal. Pairs where either completion is a Failure are skipped: those
    are excluded for a different reason. This is a property of the pair, so it
    is a single number per model rather than one per condition."""
    clusters: list[list[float]] = []
    for datapoint_scores in scores:
        if datapoint_scores.control_scores is None:
            continue
        clusters.append(
            [
                float(
                    isinstance(malicious_score, Refusal)
                    or isinstance(control_score, Refusal)
                )
                for malicious_score, control_score in zip(
                    datapoint_scores.malicious_scores,
                    datapoint_scores.control_scores,
                    strict=True,
                )
                if not isinstance(malicious_score, Failure)
                and not isinstance(control_score, Failure)
            ]
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
    model_legend: bool = True,
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

    # Each subplot: title, per-condition cluster function, whether the subplot
    # gets a control/malicious bar pair (as opposed to a single bar, for a
    # metric that is a property of the pair and so ignores the condition), — for
    # the one that gets a p value — the paired malicious-minus-control
    # difference cluster function with the alternative hypothesis of its z test
    # against 0, and the tint of its plotting area ("" for no tint).
    subplots: list[
        tuple[
            str,
            Callable[[list[DatapointScores], bool], list[list[float]]],
            bool,
            Callable[[list[DatapointScores]], list[list[float]]] | None,
            str | None,
            str,
        ]
    ] = [
        (
            "Average HealthBench score",
            _score_clusters,
            True,
            _score_diff_clusters,
            "less",
            _DETAILEDNESS_TINT,
        ),
        (
            "Fraction of statements on which<br>responses contradict each other<br>that are correct",
            _claim_correct_clusters,
            True,
            None,
            None,
            _QUALITY_TINT,
        ),
        (
            "Fraction excluded",
            lambda scores, _malicious: _excluded_clusters(scores),
            False,
            None,
            None,
            "",
        ),
    ]

    positions: list[tuple[int, int]] = [(1, 1), (1, 2), (1, 3)]

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
    _SUBPLOT_TITLE_SHIFT = 14  # gap between a subplot title and its subplot
    _SUBPLOT_TITLE_LINE_HEIGHT = 40 / 3  # the titles are three lines tall
    _SUBPLOT_TITLES_HEIGHT = (
        3 * _SUBPLOT_TITLE_LINE_HEIGHT + _SUBPLOT_TITLE_SHIFT
    )  # three lines, plus the gap
    # gap below the legend (or the caption, when there is one), a line and a
    # half of a subplot title taller than the rest of the stack's gaps
    _GAP_BELOW_LEGEND = round(10 + 1.5 * _SUBPLOT_TITLE_LINE_HEIGHT)
    _PLOT_AREA_HEIGHT = 380
    _MARGIN_BOTTOM = 90

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[subplot_title for subplot_title, *_ in subplots],
        horizontal_spacing=0.08,
    )
    # at this point the only annotations are the subplot titles; lift them a
    # little off their subplots
    for annotation in fig.layout.annotations:
        annotation.yshift = _SUBPLOT_TITLE_SHIFT

    for (row, col), (
        _,
        cluster_fn,
        paired,
        diff_cluster_fn,
        alternative,
        tint,
    ) in zip(positions, subplots, strict=True):
        if tint != "":
            fig.add_shape(
                type="rect",
                xref="x domain",
                yref="y domain",
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                fillcolor=tint,
                line_width=0,
                layer="below",
                row=row,
                col=col,
            )
        # highest error bar tip of each control/malicious bar pair, where the
        # pair's p value annotation goes
        group_tops: list[float] = [0.0] * len(results)
        bars: list[tuple[bool, str | None, str]] = (
            [
                (False, "control paraphrases", _CONTROL_COLOR),
                (True, "malicious paraphrases", _MALICIOUS_COLOR),
            ]
            if paired
            else [(False, None, _PAIR_COLOR)]
        )
        for malicious, name, color in bars:
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
                    width=None if paired else _SINGLE_BAR_WIDTH,
                    # legend-only traces share the offset group of the bar they
                    # sit next to, so that they take no slot of their own and
                    # leave the widths and positions of the bars untouched
                    offsetgroup=name if paired else "pair",
                    error_y={
                        "type": "data",
                        "array": ci95s,
                        "color": "#52514e",
                        "thickness": 1.5,
                        "width": 4,
                    },
                    legendgroup=name,
                    showlegend=paired and (row, col) == positions[0],
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

    # The tint legend needs traces of its own to hang off, so add one empty bar
    # per tint. They draw nothing (their only y is None) and share an offset
    # group with the control bars, so the bars around them keep their width and
    # position.
    for tint, edge, name in (
        (_DETAILEDNESS_TINT, _DETAILEDNESS_TINT_EDGE, "detailedness"),
        (_QUALITY_TINT, _QUALITY_TINT_EDGE, "quality"),
    ):
        fig.add_trace(
            go.Bar(
                x=[labels[0]],
                y=[None],
                name=name,
                marker={"color": tint, "line": {"color": edge, "width": 1}},
                offsetgroup="control paraphrases",
                legend="legend2",
                hoverinfo="skip",
            ),
            row=positions[0][0],
            col=positions[0][1],
        )

    title_lines: int = title.count("<br>") + 1
    title_bottom: int = _TITLE_TOP + _TITLE_LINE_HEIGHT * max(
        _RESERVED_TITLE_LINES, title_lines
    )
    title_top: int = title_bottom - _TITLE_LINE_HEIGHT * title_lines
    legend_top: int = title_bottom + 22
    # without the caption, the space it and its gap take is removed entirely
    caption_top: int = legend_top + _LEGEND_HEIGHT + (8 if model_legend else 0)
    caption_height: int = _CAPTION_HEIGHT if model_legend else 0
    margin_top: int = round(
        caption_top + caption_height + _GAP_BELOW_LEGEND + _SUBPLOT_TITLES_HEIGHT
    )
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
        # The legends sit side by side in the top margin, under the title: the
        # condition legend to the left of the tint legend. Plotly only allows
        # a legend title above (or left of) the entries, so the grader/judge
        # lines are an annotation placed just below the legends instead.
        legend={
            "orientation": "v",
            # the control trace is added first, so reversing the legend order
            # puts malicious above control without reordering the bars
            "traceorder": "reversed",
            "yanchor": "bottom",
            "y": _paper_y(legend_top + _LEGEND_HEIGHT),
            "xanchor": "right",
            "x": 0.50,
        },
        legend2={
            "orientation": "v",
            "yanchor": "bottom",
            "y": _paper_y(legend_top + _LEGEND_HEIGHT),
            "xanchor": "left",
            "x": 0.54,
        },
        width=1100,
        height=height,
        margin={"t": margin_top, "b": _MARGIN_BOTTOM},
    )
    if model_legend:
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
    # the grid is drawn above the tint rectangles (which are below the traces),
    # so that tinting a subplot does not bury its gridlines
    fig.update_yaxes(
        range=[0, 1],
        gridcolor="#d5d3c8",
        zerolinecolor="#c3c2b7",
        layer="above traces",
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")

    if html_filename is not None:
        fig.write_html(html_filename, include_plotlyjs=True)

    return fig
