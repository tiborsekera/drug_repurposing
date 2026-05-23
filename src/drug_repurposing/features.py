"""Spectral and transport-style pair features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from drug_repurposing.graph import HeterogeneousGraph


@dataclass(frozen=True)
class PairFeatures:
    """Feature vector for a candidate drug-disease pair."""

    source: str
    target: str
    heat: float
    resolvent: float
    low_frequency_distance: float
    source_degree: float
    target_degree: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.heat,
                self.resolvent,
                self.low_frequency_distance,
                self.source_degree,
                self.target_degree,
            ],
            dtype=float,
        )


def heat_kernel_matrix(laplacian: sparse.csr_matrix, tau: float = 1.0) -> np.ndarray:
    """Compute a dense heat kernel for small graphs."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    return splinalg.expm((-tau * laplacian).tocsc()).toarray()


def resolvent_matrix(operator: sparse.csr_matrix, lambda_shift: float = 1.5) -> np.ndarray:
    """Compute (lambda I - H)^-1 for small graphs."""

    if lambda_shift <= 0:
        raise ValueError("lambda_shift must be positive")
    identity = sparse.identity(operator.shape[0], format="csr")
    system = (lambda_shift * identity - operator).tocsc()
    dense_identity = np.eye(operator.shape[0])
    return splinalg.spsolve(system, dense_identity)


def low_frequency_embedding(
    laplacian: sparse.csr_matrix,
    dimensions: int = 4,
    dense_threshold: int = 256,
) -> np.ndarray:
    """Return the first non-trivial low-frequency Laplacian eigenvectors."""

    n_nodes = laplacian.shape[0]
    if n_nodes == 0:
        return np.empty((0, 0))
    if n_nodes == 1:
        return np.zeros((1, 1))

    k = min(max(dimensions + 1, 2), n_nodes - 1)
    if n_nodes <= dense_threshold:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        order = np.argsort(values)
        return vectors[:, order[1 : dimensions + 1]]

    try:
        _, vectors = splinalg.eigsh(laplacian, k=k, which="SM")
    except TypeError:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        order = np.argsort(values)
        vectors = vectors[:, order[:k]]
    except splinalg.ArpackNoConvergence:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        order = np.argsort(values)
        vectors = vectors[:, order[:k]]
    return vectors[:, 1 : dimensions + 1]


def pair_features(
    graph: HeterogeneousGraph,
    source: str,
    target: str,
    tau: float = 1.0,
    lambda_shift: float = 2.5,
    embedding_dimensions: int = 4,
) -> PairFeatures:
    """Compute MVP spectral/transport features for a candidate pair."""

    index = graph.node_index
    if source not in index:
        raise KeyError(f"Unknown source node: {source}")
    if target not in index:
        raise KeyError(f"Unknown target node: {target}")

    adjacency = graph.adjacency()
    laplacian = graph.normalized_laplacian(adjacency)
    heat = heat_kernel_matrix(laplacian, tau=tau)
    resolvent = resolvent_matrix(adjacency, lambda_shift=lambda_shift)
    embedding = low_frequency_embedding(laplacian, dimensions=embedding_dimensions)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()

    i = index[source]
    j = index[target]
    if embedding.size:
        distance = float(np.linalg.norm(embedding[i] - embedding[j]))
    else:
        distance = 0.0

    return PairFeatures(
        source=source,
        target=target,
        heat=float(heat[i, j]),
        resolvent=float(resolvent[i, j]),
        low_frequency_distance=distance,
        source_degree=float(degrees[i]),
        target_degree=float(degrees[j]),
    )
