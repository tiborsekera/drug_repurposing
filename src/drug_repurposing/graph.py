"""Small heterogeneous graph representation and sparse operator builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Node:
    """A typed biomedical KG node."""

    id: str
    type: str


@dataclass(frozen=True)
class Edge:
    """A typed, weighted KG edge."""

    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass(frozen=True)
class HeterogeneousGraph:
    """Minimal heterogeneous graph container for sparse experiments."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    directed: bool = False

    @classmethod
    def from_records(
        cls,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
        directed: bool = False,
    ) -> "HeterogeneousGraph":
        return cls(tuple(nodes), tuple(edges), directed=directed)

    @property
    def node_index(self) -> dict[str, int]:
        return {node.id: idx for idx, node in enumerate(self.nodes)}

    @property
    def node_types(self) -> dict[str, str]:
        return {node.id: node.type for node in self.nodes}

    @property
    def relations(self) -> tuple[str, ...]:
        return tuple(sorted({edge.relation for edge in self.edges}))

    def adjacency(
        self,
        relation_weights: dict[str, float] | None = None,
        include_relations: set[str] | None = None,
    ) -> sparse.csr_matrix:
        """Build a weighted adjacency matrix summed across selected relations."""

        relation_weights = relation_weights or {}
        index = self.node_index
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for edge in self.edges:
            if include_relations is not None and edge.relation not in include_relations:
                continue
            if edge.source not in index or edge.target not in index:
                raise KeyError(f"Edge references unknown node: {edge}")
            weight = edge.weight * relation_weights.get(edge.relation, 1.0)
            rows.append(index[edge.source])
            cols.append(index[edge.target])
            data.append(weight)
            if not self.directed and edge.source != edge.target:
                rows.append(index[edge.target])
                cols.append(index[edge.source])
                data.append(weight)

        shape = (len(self.nodes), len(self.nodes))
        return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()

    def normalized_laplacian(self, adjacency: sparse.csr_matrix | None = None) -> sparse.csr_matrix:
        """Return the symmetric normalized Laplacian I - D^-1/2 A D^-1/2."""

        adjacency = self.adjacency() if adjacency is None else adjacency.tocsr()
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(degrees, dtype=float)
        positive = degrees > 0
        inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
        scale = sparse.diags(inv_sqrt)
        identity = sparse.identity(adjacency.shape[0], format="csr")
        return identity - scale @ adjacency @ scale

    def without_edges(self, kept_edges: Iterable[Edge]) -> "HeterogeneousGraph":
        return HeterogeneousGraph(self.nodes, tuple(kept_edges), directed=self.directed)


def toy_biomedical_graph() -> HeterogeneousGraph:
    """Return a deterministic toy KG for examples and tests."""

    nodes = (
        Node("drug:metformin", "drug"),
        Node("drug:imatinib", "drug"),
        Node("disease:diabetes", "disease"),
        Node("disease:leukemia", "disease"),
        Node("gene:AMPK", "gene"),
        Node("gene:BCR_ABL", "gene"),
        Node("pathway:glucose", "pathway"),
    )
    edges = (
        Edge("drug:metformin", "gene:AMPK", "targets", 1.0),
        Edge("gene:AMPK", "pathway:glucose", "participates_in", 0.8),
        Edge("pathway:glucose", "disease:diabetes", "associated_with", 1.0),
        Edge("drug:imatinib", "gene:BCR_ABL", "targets", 1.0),
        Edge("gene:BCR_ABL", "disease:leukemia", "associated_with", 1.0),
        Edge("gene:AMPK", "disease:diabetes", "associated_with", 0.7),
        Edge("drug:metformin", "disease:diabetes", "indication", 1.0),
        Edge("drug:imatinib", "disease:leukemia", "indication", 1.0),
    )
    return HeterogeneousGraph.from_records(nodes, edges)
