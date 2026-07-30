"""Compare literature review sandbagging results across models in a single figure.

Usage:
    from src.plot_literature_review import plot
    fig = plot(models, results, "plots/literature_review/plot.html")

Five horizontal subplots, each a grouped malicious/control bar pair per model:
average number of papers per response, average number of claims per response,
fraction of papers that are not hallucinated, fraction of claims that are
supported by the paper they are attributed to, and fraction of responses that
are refusals.

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
"""

from collections.abc import Callable
from math import nan, sqrt
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
    html_filename: str | None = None,
) -> go.Figure:
    assert len(models) == len(results)

    labels: list[str] = [model.model.split("/")[-1] for model in models]
    if len(set(labels)) != len(labels):
        labels = [model.model for model in models]

    # (subplot title, y axis title, cluster function, whether the y axis is [0, 1])
    subplots: list[
        tuple[
            str,
            str,
            Callable[[list[EvaluatedDatapoint], bool], list[list[float]]],
            bool,
        ]
    ] = [
        (
            "Papers per response",
            "average number of papers",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _n_papers
            ),
            False,
        ),
        (
            "Claims per response",
            "average number of claims",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _n_claims
            ),
            False,
        ),
        (
            "Papers not hallucinated",
            "fraction of judged papers that exist",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _fraction_papers_not_hallucinated
            ),
            True,
        ),
        (
            "Claims supported by papers",
            "fraction of judged claims that are supported",
            lambda datapoints, malicious: _paired_clusters(
                datapoints, malicious, _fraction_claims_supported
            ),
            True,
        ),
        (
            "Refusals",
            "fraction of non-failed responses refused",
            _refusal_clusters,
            True,
        ),
    ]

    fig = make_subplots(
        rows=1,
        cols=len(subplots),
        subplot_titles=[title for title, _, _, _ in subplots],
    )

    for col, (_, y_title, cluster_fn, _) in enumerate(subplots, start=1):
        for malicious, name, color in [
            (False, "control", _CONTROL_COLOR),
            (True, "malicious", _MALICIOUS_COLOR),
        ]:
            means: list[float] = []
            ci95s: list[float] = []
            for result in results:
                mean, ci95 = _cluster_mean_and_ci95(
                    cluster_fn(result.evaluated_datapoints, malicious)
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
                    showlegend=col == 1,
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
        width=2200,
        height=500,
        margin={"t": 90},
    )
    fig.update_yaxes(range=[0, 1], gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    # counts are unbounded, so [0, 1] would clip them; the fractions stay in [0, 1]
    for col, (_, _, _, unit_range) in enumerate(subplots, start=1):
        if not unit_range:
            fig.update_yaxes(
                range=None, autorange=True, rangemode="tozero", row=1, col=col
            )

    if html_filename is not None:
        fig.write_html(html_filename, include_plotlyjs=True)

    return fig
