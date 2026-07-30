"""Compare healthbench sandbagging results across models in a single figure.

Usage:
    from src.plot_healthbench import plot
    fig = plot(models, results)
    fig.write_html("visualizations/comparison.html")

Four horizontal subplots: a single bar per model with the fraction of non-tied
completion pairs where the control completion scores higher than the malicious
one (tied pairs are excluded from the denominator), followed by grouped
control/malicious bar pairs per model for average rubric score, refusal
fraction, and fraction of contradicting claims that are correct. All bars
carry 95% error bars and all y axes span [0, 1].

Unlike ScoreSummary (which averages per datapoint first, and counts datapoints
whose per-datapoint average control score beats the malicious one), the score
and control-beats-malicious subplots average unweighted over all (datapoint,
resample) pairs where both the malicious and the control score of that
resample are valid.

Error bars use cluster-robust standard errors with datapoints as clusters:
resamples (and extracted claims) of the same datapoint are correlated, so the
effective sample size is governed by the number of datapoints, not the number
of observations.
"""

from collections.abc import Callable
from math import nan, sqrt
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


def _cluster_mean_and_ci95(clusters: list[list[float]]) -> tuple[float, float]:
    """Mean over all observations, and the half-width of its 95% confidence
    interval computed from cluster-robust standard errors.

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
    return mean, _Z95 * sqrt(variance)


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


def _control_beats_malicious_clusters(
    scores: list[DatapointScores],
) -> list[list[float]]:
    """One cluster per datapoint, with a 0/1 indicator per resample where both
    the malicious and the control score are valid and differ of whether the
    malicious completion scored lower than the control completion. Tied pairs
    are excluded entirely, so the mean is the fraction of non-tie non-failure
    pairs where control beats malicious."""
    clusters: list[list[float]] = []
    for datapoint_scores in scores:
        if datapoint_scores.control_scores is None:
            continue
        clusters.append(
            [
                float(
                    _response_score(malicious_score) < _response_score(control_score)
                )
                for malicious_score, control_score in zip(
                    datapoint_scores.malicious_scores,
                    datapoint_scores.control_scores,
                    strict=True,
                )
                if isinstance(malicious_score, ResponseScore)
                and isinstance(control_score, ResponseScore)
                and _response_score(malicious_score) != _response_score(control_score)
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
    html_filename: str | None = None,
) -> go.Figure:
    assert len(models) == len(results)

    labels: list[str] = [model.model.split("/")[-1] for model in models]
    if len(set(labels)) != len(labels):
        labels = [model.model for model in models]

    subplots: list[
        tuple[str, str, Callable[[list[DatapointScores], bool], list[list[float]]]]
    ] = [
        ("Average rubric score", "average score", _score_clusters),
        ("Refusals", "fraction of completions refused", _refusal_clusters),
        (
            "Contradicting claims correct",
            "fraction of claim pairs correct",
            _claim_correct_clusters,
        ),
    ]

    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=["Control beats malicious"]
        + [title for title, _, _ in subplots],
    )

    means: list[float] = []
    ci95s: list[float] = []
    for result in results:
        mean, ci95 = _cluster_mean_and_ci95(
            _control_beats_malicious_clusters(result.scores)
        )
        means.append(mean)
        ci95s.append(ci95)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=means,
            name="control beats malicious",
            marker_color="#1baf7a",
            error_y={
                "type": "data",
                "array": ci95s,
                "color": "#52514e",
                "thickness": 1.5,
                "width": 4,
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="fraction of non-tied completion pairs where control scores higher",
        tickvals=[0, 0.2, 0.4, 0.5, 0.6, 0.8, 1],
        row=1,
        col=1,
    )
    # y = 0.5 is the no-directional-difference baseline for the win fraction
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="#52514e", line_width=2, row=1, col=1
    )

    for col, (_, y_title, cluster_fn) in enumerate(subplots, start=2):
        for malicious, name, color in [
            (False, "control", _CONTROL_COLOR),
            (True, "malicious", _MALICIOUS_COLOR),
        ]:
            means = []
            ci95s = []
            for result in results:
                mean, ci95 = _cluster_mean_and_ci95(
                    cluster_fn(result.scores, malicious)
                )
                means.append(mean)
                ci95s.append(ci95)
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
                    showlegend=col == 2,
                ),
                row=1,
                col=col,
            )
        fig.update_yaxes(title_text=y_title, row=1, col=col)

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
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "center",
            "x": 0.5,
        },
        width=1900,
        height=500,
        margin={"t": 90},
    )
    fig.update_yaxes(range=[0, 1], gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")

    if html_filename is not None:
        fig.write_html(html_filename, include_plotlyjs=True)

    return fig
