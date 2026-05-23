"""Disorder models and robustness summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Iterable, overload

import numpy as np

from drug_repurposing.graph import Edge, HeterogeneousGraph

EdgeRecord = tuple[str, str, str, float]


@dataclass(frozen=True)
class RobustnessSummary:
    """Mean/variance robust scoring summary over graph perturbations."""

    mean_score: float
    variance: float
    robust_score: float
    n_realizations: int
    scores: tuple[float, ...]


@overload
def edge_dropout(
    graph: HeterogeneousGraph,
    dropout_probability: float | None = None,
    seed: int | None = None,
    protected_relations: set[str] | None = None,
    drop_probability: float | None = None,
) -> HeterogeneousGraph:
    ...


@overload
def edge_dropout(
    graph: Sequence[EdgeRecord],
    dropout_probability: float | None = None,
    seed: int | None = None,
    protected_relations: set[str] | None = None,
    drop_probability: float | None = None,
) -> list[EdgeRecord]:
    ...


def edge_dropout(
    graph: HeterogeneousGraph | Sequence[EdgeRecord],
    dropout_probability: float | None = None,
    seed: int | None = None,
    protected_relations: set[str] | None = None,
    drop_probability: float | None = None,
) -> HeterogeneousGraph | list[EdgeRecord]:
    """Randomly remove edges, optionally preserving selected relations."""

    if dropout_probability is None:
        dropout_probability = drop_probability
    if dropout_probability is None:
        raise TypeError("dropout_probability or drop_probability is required")
    if not 0 <= dropout_probability <= 1:
        raise ValueError("dropout_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    protected_relations = protected_relations or set()
    if isinstance(graph, HeterogeneousGraph):
        kept: list[Edge] = []
        for edge in graph.edges:
            if edge.relation in protected_relations or rng.random() >= dropout_probability:
                kept.append(edge)
        return graph.without_edges(kept)

    kept_records: list[EdgeRecord] = []
    for edge in graph:
        _, relation, _, _ = edge
        if relation in protected_relations or rng.random() >= dropout_probability:
            kept_records.append(edge)
    return kept_records


def relation_dropout(graph: HeterogeneousGraph, relations: Iterable[str]) -> HeterogeneousGraph:
    """Remove all edges belonging to the supplied relation names."""

    removed = set(relations)
    return graph.without_edges(edge for edge in graph.edges if edge.relation not in removed)


def weight_noise(
    graph: HeterogeneousGraph,
    sigma: float,
    seed: int | None = None,
    minimum_weight: float = 0.0,
) -> HeterogeneousGraph:
    """Apply multiplicative log-normal edge-weight noise."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(seed)
    noisy_edges = []
    for edge in graph.edges:
        factor = float(rng.lognormal(mean=0.0, sigma=sigma))
        noisy_edges.append(
            Edge(
                edge.source,
                edge.target,
                edge.relation,
                max(minimum_weight, edge.weight * factor),
            )
        )
    return graph.without_edges(noisy_edges)


def summarize_robustness(
    graph: HeterogeneousGraph,
    scorer: Callable[[HeterogeneousGraph], float],
    perturb: Callable[[HeterogeneousGraph, int], HeterogeneousGraph],
    n_realizations: int = 16,
    beta: float = 1.0,
    seed: int = 0,
) -> RobustnessSummary:
    """Score perturbation realizations and compute mean - beta * variance."""

    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive")

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_realizations):
        realization_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        perturbed = perturb(graph, realization_seed)
        scores.append(float(scorer(perturbed)))

    score_array = np.asarray(scores, dtype=float)
    mean_score = float(score_array.mean())
    variance = float(score_array.var())
    return RobustnessSummary(
        mean_score=mean_score,
        variance=variance,
        robust_score=float(mean_score - beta * variance),
        n_realizations=n_realizations,
        scores=tuple(scores),
    )


def summarize_score_robustness(
    edges: Sequence[EdgeRecord],
    pairs: Sequence[tuple[str, str]],
    score_fn: Callable[[Sequence[EdgeRecord]], dict[tuple[str, str], float]],
    perturbation: Callable[..., Sequence[EdgeRecord]],
    perturbation_kwargs: dict[str, object] | None = None,
    seeds: Sequence[int] = tuple(range(16)),
    beta: float = 1.0,
) -> list[dict[str, object]]:
    """Summarize pair scores over tuple-edge perturbation realizations."""

    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    perturbation_kwargs = perturbation_kwargs or {}
    pair_scores: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    for seed in seeds:
        perturbed_edges = perturbation(edges, seed=seed, **perturbation_kwargs)
        scores = score_fn(perturbed_edges)
        for pair in pairs:
            pair_scores[pair].append(float(scores[pair]))

    rows: list[dict[str, object]] = []
    for pair, scores in pair_scores.items():
        score_array = np.asarray(scores, dtype=float)
        mean_score = float(score_array.mean())
        score_variance = float(score_array.var())
        rows.append(
            {
                "pair": pair,
                "mean_score": mean_score,
                "score_variance": score_variance,
                "score_std": float(score_array.std()),
                "min_score": float(score_array.min()),
                "max_score": float(score_array.max()),
                "robust_score": float(mean_score - beta * score_variance),
                "n_perturbations": len(seeds),
            }
        )
    return rows
