"""Convenience feature API for tuple-based graph experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from drug_repurposing.features import heat_kernel_matrix

EdgeRecord = tuple[str, str, str, float]


def build_weighted_adjacency(
    edges: Sequence[EdgeRecord],
    node_order: Sequence[str] | None = None,
    relation_weights: dict[str, float] | None = None,
    undirected: bool = True,
) -> sparse.csr_matrix:
    """Build a sparse weighted adjacency from `(source, relation, target, weight)` records."""

    relation_weights = relation_weights or {}
    if node_order is None:
        node_order = sorted({source for source, _, _, _ in edges} | {target for _, _, target, _ in edges})
    node_index = {node: idx for idx, node in enumerate(node_order)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for source, relation, target, weight in edges:
        edge_weight = float(weight) * relation_weights.get(relation, 1.0)
        rows.append(node_index[source])
        cols.append(node_index[target])
        data.append(edge_weight)
        if undirected and source != target:
            rows.append(node_index[target])
            cols.append(node_index[source])
            data.append(edge_weight)

    shape = (len(node_order), len(node_order))
    return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()


def normalized_laplacian(adjacency: sparse.spmatrix) -> sparse.csr_matrix:
    """Return the symmetric normalized Laplacian I - D^-1/2 A D^-1/2."""

    adjacency = adjacency.tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees, dtype=float)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    scale = sparse.diags(inv_sqrt)
    identity = sparse.identity(adjacency.shape[0], format="csr")
    return identity - scale @ adjacency @ scale


def heat_kernel_pair_features(
    laplacian: sparse.spmatrix,
    node_index: dict[str, int],
    pairs: Sequence[tuple[str, str]],
    taus: Sequence[float],
) -> np.ndarray:
    """Return heat-kernel pair entries with shape `(n_pairs, n_taus)`."""

    features = np.zeros((len(pairs), len(taus)), dtype=float)
    for tau_idx, tau in enumerate(taus):
        kernel = heat_kernel_matrix(laplacian.tocsr(), tau=float(tau))
        for pair_idx, (source, target) in enumerate(pairs):
            features[pair_idx, tau_idx] = kernel[node_index[source], node_index[target]]
    return features
