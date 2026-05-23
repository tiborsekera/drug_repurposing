"""Transparent baseline scorers for drug-disease candidate ranking."""

from __future__ import annotations

import numpy as np

from drug_repurposing.features import pair_features
from drug_repurposing.graph import HeterogeneousGraph


def spectral_score(graph: HeterogeneousGraph, drug_id: str, disease_id: str) -> float:
    """Simple spectral transport score used as an interpretable MVP baseline."""

    features = pair_features(graph, drug_id, disease_id)
    return float(features.heat + 0.25 * features.resolvent - features.low_frequency_distance)


def degree_product_score(graph: HeterogeneousGraph, drug_id: str, disease_id: str) -> float:
    """Degree-bias baseline for detecting popularity leakage."""

    adjacency = graph.adjacency()
    index = graph.node_index
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return float(degrees[index[drug_id]] * degrees[index[disease_id]])


def common_neighbor_score(graph: HeterogeneousGraph, drug_id: str, disease_id: str) -> float:
    """Classical network proximity proxy based on common neighbors."""

    adjacency = graph.adjacency().astype(bool).astype(int)
    index = graph.node_index
    drug_neighbors = adjacency[index[drug_id]].toarray().ravel()
    disease_neighbors = adjacency[index[disease_id]].toarray().ravel()
    return float(np.dot(drug_neighbors, disease_neighbors))
