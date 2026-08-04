"""Compare literature review sandbagging results across models in a single figure.

Usage:
    from src.plot_literature_review import plot
    fig = plot(
        models, results, title, claim_extractor, correctness_judge, refusal_judge,
    )
    fig.write_html("plots/literature_review/plot.html")

Five subplots: a 2x2 grid of per model bars, one row per measure of how much a
response says — papers, then claims — with, in each row, the average of the
measure per response and the fraction of the measure that is good (papers that
are not hallucinated, claims that are supported). To the right of the grid,
vertically centered, is a fraction of excluded response pairs subplot. The
averages and fractions of X are grouped malicious/control bar pairs; the
exclusion fraction is a property of the pair, so it is a single bar per model.
With papers_only=True, only the papers row and the fraction excluded subplot are
drawn, in a single row laid out like src/plot_healthbench.py.

A tint behind the plotting area says which family a subplot's metric belongs to:
how extensive the response is, or how good what it says is. The exclusion
fraction is neither, so that subplot keeps the plain background. A second legend,
beside the control/malicious one, is the key to the two tints.

Except for the exclusion subplot, every average is taken over response pairs:
the malicious and control response of the ith resample of the jth datapoint, for
all (i, j) where neither of the two is a Failure or a Refusal. The exclusion
subplot averages, over the pairs where neither response is a Failure, a 0/1
indicator of whether the pair is excluded because the malicious or the control
response is a Refusal. That is a property of the pair rather than of one
condition, so it is drawn as a single bar per model.

The two fraction-of-X subplots average per-response fractions (not pooled
counts), and skip responses whose denominator is 0. Denominators exclude judge
failures: the fraction of non-hallucinated papers is over the papers the paper
existence judge ruled on, and the fraction of supported claims is over the
claims the claim correctness judge ruled on.

Error bars use cluster-robust standard errors with datapoints as clusters:
resamples of the same datapoint are correlated, so the effective sample size is
governed by the number of datapoints, not the number of observations.

P values are annotated above every subplot but the exclusion one, and come from
one-sided cluster-robust z tests with the same clustering, taken on the side
that sandbagging would push the metric to: the tested quantity is the per-pair
paired difference (malicious minus control), the null is 0 and the alternative
is that the difference is negative.
"""

from collections.abc import Callable
from dataclasses import dataclass
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
# subplots whose metric is a property of a response pair get a single neutral
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


# The per-response values the subplots are built out of. A value of None means
# the response is excluded from the metric.


def _n_papers(response: EvaluatedResponse) -> float | None:
    return response.summary.n_papers


def _n_claims(response: EvaluatedResponse) -> float | None:
    return response.summary.n_claims


def _fraction_papers_not_hallucinated(response: EvaluatedResponse) -> float | None:
    """Fraction of the papers the existence judge ruled on that exist. None
    (i.e. this response is excluded) if the judge ruled on no paper."""
    summary: ResponseSummary = response.summary
    n_judged: int = summary.n_papers - summary.n_paper_judge_failed
    if n_judged == 0:
        return None
    return summary.n_correct_papers / n_judged


def _fraction_claims_supported(response: EvaluatedResponse) -> float | None:
    """Fraction of the claims the correctness judge ruled on that are supported
    by the paper they are attributed to. None (i.e. this response is excluded)
    if the judge ruled on no claim."""
    summary: ResponseSummary = response.summary
    n_judged: int = summary.n_existing_paper_claims - summary.n_claim_judge_failed
    if n_judged == 0:
        return None
    return summary.n_supported_paper_claims / n_judged


def _paired_clusters(
    evaluated_datapoints: list[EvaluatedDatapoint],
    malicious: bool,
    value: Callable[[EvaluatedResponse], float | None],
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
            response_value: float | None = value(response)
            if response_value is None:
                continue
            cluster.append(response_value)
        clusters.append(cluster)
    return clusters


def _paired_diff_clusters(
    evaluated_datapoints: list[EvaluatedDatapoint],
    value: Callable[[EvaluatedResponse], float | None],
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
            malicious_value: float | None = value(pair.malicious)
            control_value: float | None = value(pair.control)
            if malicious_value is None or control_value is None:
                continue
            cluster.append(malicious_value - control_value)
        clusters.append(cluster)
    return clusters


def _excluded_clusters(
    evaluated_datapoints: list[EvaluatedDatapoint],
) -> list[list[float]]:
    """One cluster per datapoint, with a 0/1 indicator per response pair of
    whether the pair is excluded because the malicious or the control response
    is a refusal. Pairs where either response is a Failure are skipped: those
    are excluded for a different reason. A pair whose malicious response is a
    Refusal has no control response (it is never generated) and still counts as
    excluded. This is a property of the pair, so it is a single number per model
    rather than one per condition."""
    clusters: list[list[float]] = []
    for datapoint in evaluated_datapoints:
        cluster: list[float] = []
        for pair in datapoint.evaluated_response_pairs:
            if isinstance(pair.malicious, Failure) or isinstance(pair.control, Failure):
                continue
            cluster.append(
                float(
                    isinstance(pair.malicious, Refusal)
                    or isinstance(pair.control, Refusal)
                )
            )
        clusters.append(cluster)
    return clusters


@dataclass(frozen=True)
class _Subplot:
    """One subplot of the figure.

    `clusters` gives the clusters averaged into the bars, as a function of the
    evaluated datapoints and of whether the malicious condition is wanted;
    single-bar subplots ignore that second argument. `paired` says whether the
    subplot gets a control/malicious bar pair, as opposed to a single bar for a
    metric that is a property of the pair. `unit_range` says whether the y axis
    is fixed to [0, 1]. `p_value_clusters` gives the clusters of the paired
    differences that the annotated one-sided z test tests against 0, and is None
    for a subplot with no p value."""

    title: str
    clusters: Callable[[list[EvaluatedDatapoint], bool], list[list[float]]]
    paired: bool
    unit_range: bool
    p_value_clusters: Callable[[list[EvaluatedDatapoint]], list[list[float]]] | None
    position: tuple[int, int]
    tint: str


def plot(
    models: list[Model],
    results: list[ExperimentResult],
    title: str,
    claim_extractor: str,
    correctness_judge: str,
    refusal_judge: str,
    model_name_map: dict[str, str] | None = None,
    html_filename: str | None = None,
    papers_only: bool = False,
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

    papers_subplot = _Subplot(
        title="Papers per response",
        clusters=lambda datapoints, malicious: _paired_clusters(
            datapoints, malicious, _n_papers
        ),
        paired=True,
        unit_range=False,
        p_value_clusters=lambda datapoints: _paired_diff_clusters(
            datapoints, _n_papers
        ),
        position=(1, 1),
        tint=_DETAILEDNESS_TINT,
    )
    not_hallucinated_subplot = _Subplot(
        title="Fraction papers not hallucinated",
        clusters=lambda datapoints, malicious: _paired_clusters(
            datapoints, malicious, _fraction_papers_not_hallucinated
        ),
        paired=True,
        unit_range=True,
        p_value_clusters=lambda datapoints: _paired_diff_clusters(
            datapoints, _fraction_papers_not_hallucinated
        ),
        position=(1, 2),
        tint=_QUALITY_TINT,
    )
    excluded_subplot = _Subplot(
        title="Fraction excluded",
        clusters=lambda datapoints, _malicious: _excluded_clusters(datapoints),
        paired=False,
        unit_range=True,
        p_value_clusters=None,
        position=(1, 3),
        tint="",
    )
    subplots: list[_Subplot]
    if papers_only:
        subplots = [
            papers_subplot,
            not_hallucinated_subplot,
            excluded_subplot,
        ]
    else:
        subplots = [
            papers_subplot,
            not_hallucinated_subplot,
            excluded_subplot,
            _Subplot(
                title="Claims per response",
                clusters=lambda datapoints, malicious: _paired_clusters(
                    datapoints, malicious, _n_claims
                ),
                paired=True,
                unit_range=False,
                p_value_clusters=lambda datapoints: _paired_diff_clusters(
                    datapoints, _n_claims
                ),
                position=(2, 1),
                tint=_DETAILEDNESS_TINT,
            ),
            _Subplot(
                title="Fraction claims supported by papers",
                clusters=lambda datapoints, malicious: _paired_clusters(
                    datapoints, malicious, _fraction_claims_supported
                ),
                paired=True,
                unit_range=True,
                p_value_clusters=lambda datapoints: _paired_diff_clusters(
                    datapoints, _fraction_claims_supported
                ),
                position=(2, 2),
                tint=_QUALITY_TINT,
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
    _SUBPLOT_TITLE_SHIFT = 14  # gap between a subplot title and its subplot
    _SUBPLOT_TITLES_HEIGHT = 24 + _SUBPLOT_TITLE_SHIFT  # one line, plus the gap
    _CAPTION_GAP = 26  # between the caption (or legend) and the subplot titles
    _ROW_HEIGHT = 380
    # gap between two grid rows: the upper row's x tick labels plus the lower
    # row's subplot titles
    _ROW_GAP = 168
    _ROWS = 1 if papers_only else 2
    _PLOT_AREA_HEIGHT = _ROWS * _ROW_HEIGHT + (_ROWS - 1) * _ROW_GAP
    _MARGIN_BOTTOM = 90

    if papers_only:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[subplot.title for subplot in subplots],
            horizontal_spacing=0.06,
        )
    else:
        # the exclusion subplot spans the whole grid height
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{}, {}, {"rowspan": 2}],
                [{}, {}, None],
            ],
            subplot_titles=[subplot.title for subplot in subplots],
            horizontal_spacing=0.06,
            vertical_spacing=_ROW_GAP / _PLOT_AREA_HEIGHT,
        )
    # at this point the only annotations are the subplot titles; lift them a
    # little off their subplots
    for annotation in fig.layout.annotations:
        annotation.yshift = _SUBPLOT_TITLE_SHIFT

    for subplot in subplots:
        row, col = subplot.position
        if subplot.tint != "":
            fig.add_shape(
                type="rect",
                xref="x domain",
                yref="y domain",
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                fillcolor=subplot.tint,
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
                (False, "control", _CONTROL_COLOR),
                (True, "malicious", _MALICIOUS_COLOR),
            ]
            if subplot.paired
            else [(False, None, _PAIR_COLOR)]
        )
        for malicious, name, color in bars:
            means: list[float] = []
            ci95s: list[float] = []
            for i, result in enumerate(results):
                mean, se = _cluster_mean_and_se(
                    subplot.clusters(result.evaluated_datapoints, malicious)
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
                    width=None if subplot.paired else _SINGLE_BAR_WIDTH,
                    # legend-only traces share the offset group of the bar they
                    # sit next to, so that they take no slot of their own and
                    # leave the widths and positions of the bars untouched
                    offsetgroup=name if subplot.paired else "pair",
                    error_y={
                        "type": "data",
                        "array": ci95s,
                        "color": "#52514e",
                        "thickness": 1.5,
                        "width": 4,
                    },
                    legendgroup=name,
                    showlegend=subplot.paired
                    and subplot.position == subplots[0].position,
                ),
                row=row,
                col=col,
            )
        if subplot.p_value_clusters is None:
            continue
        for label, group_top, result in zip(labels, group_tops, results, strict=True):
            mean, se = _cluster_mean_and_se(
                subplot.p_value_clusters(result.evaluated_datapoints)
            )
            _add_p_annotation(
                fig,
                label,
                group_top,
                _z_test_p(mean, null=0.0, se=se, alternative="less"),
                row=row,
                col=col,
                y_max=1.0 if subplot.unit_range else None,
            )

    # The tint legend needs traces of its own to hang off, so add one empty bar
    # per tint. They draw nothing (their only y is None) and share an offset
    # group with the control bars, so the bars around them keep their width and
    # position.
    for tint, edge, tint_name in (
        (_DETAILEDNESS_TINT, _DETAILEDNESS_TINT_EDGE, "detailedness"),
        (_QUALITY_TINT, _QUALITY_TINT_EDGE, "quality"),
    ):
        first_row, first_col = subplots[0].position
        fig.add_trace(
            go.Bar(
                x=[labels[0]],
                y=[None],
                name=tint_name,
                marker={"color": tint, "line": {"color": edge, "width": 1}},
                offsetgroup="control",
                legend="legend2",
                hoverinfo="skip",
            ),
            row=first_row,
            col=first_col,
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
    margin_top: int = (
        caption_top + caption_height + _CAPTION_GAP + _SUBPLOT_TITLES_HEIGHT
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
        # a legend title above (or left of) the entries, so the extractor/judge
        # lines are an annotation placed just below the legends instead.
        legend={
            "orientation": "v",
            # the control trace is added first, so reversing the legend order
            # puts malicious above control without reordering the bars
            "traceorder": "reversed",
            "yanchor": "bottom",
            "y": _paper_y(legend_top + _LEGEND_HEIGHT),
            "xanchor": "right",
            "x": 0.47,
        },
        legend2={
            "orientation": "v",
            "yanchor": "bottom",
            "y": _paper_y(legend_top + _LEGEND_HEIGHT),
            "xanchor": "left",
            "x": 0.51,
        },
        # the same width per column as src/plot_healthbench.py, so that the
        # subplot titles still fit on one line
        width=1100,
        height=height,
        margin={"t": margin_top, "b": _MARGIN_BOTTOM},
    )
    if model_legend:
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
    # the grid is drawn above the tint rectangles (which are below the traces),
    # so that tinting a subplot does not bury its gridlines
    fig.update_yaxes(
        range=[0, 1],
        gridcolor="#d5d3c8",
        zerolinecolor="#c3c2b7",
        layer="above traces",
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    # counts are unbounded, so [0, 1] would clip them; the fractions stay in [0, 1]
    for subplot in subplots:
        if not subplot.unit_range:
            row, col = subplot.position
            fig.update_yaxes(
                range=None, autorange=True, rangemode="tozero", row=row, col=col
            )

    # In the full layout the exclusion subplot spans every grid row; shrink it
    # to one row's height, vertically centered, and move its subplot title down
    # with it.
    if not papers_only:
        excluded_row, excluded_col = excluded_subplot.position
        row_fraction: float = _ROW_HEIGHT / _PLOT_AREA_HEIGHT
        excluded_domain: list[float] = [
            (1 - row_fraction) / 2,
            (1 + row_fraction) / 2,
        ]
        fig.update_yaxes(domain=excluded_domain, row=excluded_row, col=excluded_col)
        for annotation in fig.layout.annotations:
            if annotation.text == excluded_subplot.title:
                annotation.y = excluded_domain[1]

    if html_filename is not None:
        fig.write_html(html_filename, include_plotlyjs=True)

    return fig
