"""Executable toy pipeline for the first spectral-disorder MVP."""

from __future__ import annotations

import json

from drug_repurposing.features import pair_features
from drug_repurposing.graph import HeterogeneousGraph, toy_biomedical_graph
from drug_repurposing.perturbations import edge_dropout, summarize_robustness


def metformin_diabetes_score(graph: HeterogeneousGraph) -> float:
    features = pair_features(graph, "drug:metformin", "disease:diabetes")
    return features.heat + 0.25 * features.resolvent - features.low_frequency_distance


def run() -> dict[str, object]:
    graph = toy_biomedical_graph()
    features = pair_features(graph, "drug:metformin", "disease:diabetes")
    robustness = summarize_robustness(
        graph,
        scorer=metformin_diabetes_score,
        perturb=lambda g, seed: edge_dropout(
            g,
            dropout_probability=0.2,
            seed=seed,
            protected_relations={"indication"},
        ),
        n_realizations=8,
        beta=1.0,
        seed=42,
    )
    return {
        "pair": [features.source, features.target],
        "features": {
            "heat": features.heat,
            "resolvent": features.resolvent,
            "low_frequency_distance": features.low_frequency_distance,
            "source_degree": features.source_degree,
            "target_degree": features.target_degree,
        },
        "robustness": {
            "mean_score": robustness.mean_score,
            "variance": robustness.variance,
            "robust_score": robustness.robust_score,
            "n_realizations": robustness.n_realizations,
        },
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
