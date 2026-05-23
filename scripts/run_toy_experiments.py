"""Run deterministic toy drug-repurposing experiments.

The script uses the repository's small heterogeneous graph APIs to score every
toy drug-disease candidate, then measures score stability under edge dropout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drug_repurposing.features import PairFeatures, pair_features
from drug_repurposing.graph import HeterogeneousGraph, toy_biomedical_graph
from drug_repurposing.perturbations import edge_dropout, summarize_robustness

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "toy_experiments"
DEFAULT_DROPOUT_PROBABILITIES = (0.0, 0.1, 0.2, 0.35, 0.5)
DEFAULT_REALIZATIONS = 16
DEFAULT_BETA = 1.0
DEFAULT_SEED = 20260523


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    source: str
    target: str
    is_known_indication: bool


@dataclass(frozen=True)
class FeatureRow:
    pair_id: str
    source: str
    target: str
    source_type: str
    target_type: str
    is_known_indication: bool
    heat: float
    resolvent: float
    low_frequency_distance: float
    source_degree: float
    target_degree: float
    nominal_score: float


@dataclass(frozen=True)
class RobustnessRow:
    pair_id: str
    source: str
    target: str
    dropout_probability: float
    protected_relations: str
    beta: float
    seed: int
    n_realizations: int
    mean_score: float
    variance: float
    robust_score: float
    min_score: float
    max_score: float


def candidate_pairs(graph: HeterogeneousGraph) -> tuple[CandidatePair, ...]:
    node_types = graph.node_types
    drugs = sorted(node.id for node in graph.nodes if node.type == "drug")
    diseases = sorted(node.id for node in graph.nodes if node.type == "disease")
    known_indications = {
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.relation == "indication"
    } | {
        (edge.target, edge.source)
        for edge in graph.edges
        if edge.relation == "indication"
    }

    pairs: list[CandidatePair] = []
    for drug in drugs:
        for disease in diseases:
            if node_types[drug] != "drug" or node_types[disease] != "disease":
                raise ValueError(f"Unexpected candidate node types: {drug}, {disease}")
            pairs.append(
                CandidatePair(
                    pair_id=f"{drug}__{disease}".replace(":", "_"),
                    source=drug,
                    target=disease,
                    is_known_indication=(drug, disease) in known_indications,
                )
            )
    return tuple(pairs)


def nominal_score(features: PairFeatures) -> float:
    return features.heat + 0.25 * features.resolvent - features.low_frequency_distance


def graph_signature(graph: HeterogeneousGraph) -> tuple[tuple[str, str, str, float], ...]:
    return tuple(
        sorted(
            (edge.source, edge.relation, edge.target, float(edge.weight))
            for edge in graph.edges
        )
    )


def feature_rows(graph: HeterogeneousGraph, pairs: Iterable[CandidatePair]) -> tuple[FeatureRow, ...]:
    node_types = graph.node_types
    rows: list[FeatureRow] = []
    for candidate in pairs:
        features = pair_features(graph, candidate.source, candidate.target)
        rows.append(
            FeatureRow(
                pair_id=candidate.pair_id,
                source=features.source,
                target=features.target,
                source_type=node_types[features.source],
                target_type=node_types[features.target],
                is_known_indication=candidate.is_known_indication,
                heat=features.heat,
                resolvent=features.resolvent,
                low_frequency_distance=features.low_frequency_distance,
                source_degree=features.source_degree,
                target_degree=features.target_degree,
                nominal_score=nominal_score(features),
            )
        )
    return tuple(rows)


def robustness_rows(
    graph: HeterogeneousGraph,
    pairs: Iterable[CandidatePair],
    dropout_probabilities: Iterable[float],
    n_realizations: int,
    beta: float,
    seed: int,
    protected_relations: set[str],
) -> tuple[RobustnessRow, ...]:
    rows: list[RobustnessRow] = []
    protected_label = "|".join(sorted(protected_relations))
    for candidate_index, candidate in enumerate(pairs):
        score_cache: dict[tuple[tuple[str, str, str, float], ...], float] = {}

        def scorer(perturbed_graph: HeterogeneousGraph) -> float:
            signature = graph_signature(perturbed_graph)
            if signature not in score_cache:
                features = pair_features(perturbed_graph, candidate.source, candidate.target)
                score_cache[signature] = nominal_score(features)
            return score_cache[signature]

        for dropout_index, dropout_probability in enumerate(dropout_probabilities):
            sweep_seed = seed + candidate_index * 1000 + dropout_index
            summary = summarize_robustness(
                graph,
                scorer=scorer,
                perturb=lambda g, realization_seed: edge_dropout(
                    g,
                    dropout_probability=dropout_probability,
                    seed=realization_seed,
                    protected_relations=protected_relations,
                ),
                n_realizations=n_realizations,
                beta=beta,
                seed=sweep_seed,
            )
            rows.append(
                RobustnessRow(
                    pair_id=candidate.pair_id,
                    source=candidate.source,
                    target=candidate.target,
                    dropout_probability=float(dropout_probability),
                    protected_relations=protected_label,
                    beta=float(beta),
                    seed=sweep_seed,
                    n_realizations=summary.n_realizations,
                    mean_score=summary.mean_score,
                    variance=summary.variance,
                    robust_score=summary.robust_score,
                    min_score=float(min(summary.scores)),
                    max_score=float(max(summary.scores)),
                )
            )
    return tuple(rows)


def write_csv(path: Path, rows: Iterable[object]) -> None:
    materialized_rows = [asdict(row) for row in rows]
    if not materialized_rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized_rows[0]))
        writer.writeheader()
        writer.writerows(materialized_rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def metrics_payload(
    graph: HeterogeneousGraph,
    pairs: tuple[CandidatePair, ...],
    features: tuple[FeatureRow, ...],
    robustness: tuple[RobustnessRow, ...],
    dropout_probabilities: tuple[float, ...],
    n_realizations: int,
    beta: float,
    seed: int,
) -> dict[str, object]:
    top_nominal = max(features, key=lambda row: (row.nominal_score, row.pair_id))
    top_by_dropout = {}
    for dropout_probability in dropout_probabilities:
        rows = [
            row
            for row in robustness
            if row.dropout_probability == float(dropout_probability)
        ]
        top = max(rows, key=lambda row: (row.robust_score, row.pair_id))
        top_by_dropout[f"{dropout_probability:.2f}"] = {
            "pair_id": top.pair_id,
            "source": top.source,
            "target": top.target,
            "robust_score": top.robust_score,
            "mean_score": top.mean_score,
            "variance": top.variance,
        }

    return {
        "experiment": "deterministic_toy_spectral_dropout",
        "seed": seed,
        "beta": beta,
        "n_realizations": n_realizations,
        "dropout_probabilities": list(dropout_probabilities),
        "protected_relations": ["indication"],
        "graph": {
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "relations": list(graph.relations),
        },
        "n_candidate_pairs": len(pairs),
        "n_feature_rows": len(features),
        "n_robustness_rows": len(robustness),
        "score_formula": "heat + 0.25 * resolvent - low_frequency_distance",
        "top_nominal_pair": {
            "pair_id": top_nominal.pair_id,
            "source": top_nominal.source,
            "target": top_nominal.target,
            "nominal_score": top_nominal.nominal_score,
        },
        "top_robust_pair_by_dropout": top_by_dropout,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for features.csv, robustness.csv, and metrics.json. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=DEFAULT_REALIZATIONS,
        help="Number of edge-dropout realizations per pair and dropout probability.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Variance penalty for robust_score = mean_score - beta * variance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed used to derive deterministic sweep seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_realizations <= 0:
        raise ValueError("--n-realizations must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = toy_biomedical_graph()
    pairs = candidate_pairs(graph)
    dropout_probabilities = tuple(DEFAULT_DROPOUT_PROBABILITIES)
    features = feature_rows(graph, pairs)
    robustness = robustness_rows(
        graph=graph,
        pairs=pairs,
        dropout_probabilities=dropout_probabilities,
        n_realizations=args.n_realizations,
        beta=args.beta,
        seed=args.seed,
        protected_relations={"indication"},
    )

    write_csv(output_dir / "features.csv", features)
    write_csv(output_dir / "robustness.csv", robustness)
    write_json(
        output_dir / "metrics.json",
        metrics_payload(
            graph=graph,
            pairs=pairs,
            features=features,
            robustness=robustness,
            dropout_probabilities=dropout_probabilities,
            n_realizations=args.n_realizations,
            beta=args.beta,
            seed=args.seed,
        ),
    )

    print(f"Wrote toy experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
