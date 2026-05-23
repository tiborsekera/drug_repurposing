"""Materialize EXP-001 and EXP-002 artifacts from the toy graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drug_repurposing.features import pair_features
from drug_repurposing.graph import toy_biomedical_graph


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    graph = toy_biomedical_graph()
    exp001 = REPO_ROOT / "artifacts" / "EXP-001"
    exp002 = REPO_ROOT / "artifacts" / "EXP-002"
    exp001.mkdir(parents=True, exist_ok=True)
    exp002.mkdir(parents=True, exist_ok=True)

    entities = pd.DataFrame([{"node_id": node.id, "node_type": node.type} for node in graph.nodes])
    edges = pd.DataFrame(
        [
            {"source": edge.source, "target": edge.target, "relation": edge.relation, "weight": edge.weight}
            for edge in graph.edges
        ]
    )
    relations = pd.DataFrame({"relation": sorted(edges["relation"].unique())})
    adjacency = graph.adjacency()
    laplacian = graph.normalized_laplacian(adjacency)

    entities.to_csv(exp001 / "entity_table.csv", index=False)
    edges.to_csv(exp001 / "edge_table.csv", index=False)
    relations.to_csv(exp001 / "relation_table.csv", index=False)
    sparse.save_npz(exp001 / "adjacency.npz", adjacency)
    write_json(
        exp001 / "config.json",
        {
            "experiment_id": "EXP-001",
            "graph": "toy_biomedical_graph",
            "directed": graph.directed,
        },
    )
    write_json(
        exp001 / "metrics.json",
        {
            "node_count": int(len(graph.nodes)),
            "edge_count": int(len(graph.edges)),
            "relation_count": int(len(relations)),
            "isolated_node_count": int(np.sum(np.asarray(adjacency.sum(axis=1)).ravel() == 0)),
            "laplacian_trace": float(laplacian.diagonal().sum()),
        },
    )
    (exp001 / "run.md").write_text(
        "# EXP-001\n\nToy graph schema and sparse adjacency materialized for API validation.\n"
    )

    drugs = sorted(node.id for node in graph.nodes if node.type == "drug")
    diseases = sorted(node.id for node in graph.nodes if node.type == "disease")
    rows = []
    for drug in drugs:
        for disease in diseases:
            features = pair_features(graph, drug, disease)
            rows.append(
                {
                    "drug_id": drug,
                    "disease_id": disease,
                    "heat": features.heat,
                    "resolvent": features.resolvent,
                    "low_frequency_distance": features.low_frequency_distance,
                    "source_degree": features.source_degree,
                    "target_degree": features.target_degree,
                    "score": features.heat + 0.25 * features.resolvent - features.low_frequency_distance,
                }
            )
    feature_frame = pd.DataFrame(rows)
    feature_frame.to_csv(exp002 / "pair_features.csv", index=False)
    write_json(
        exp002 / "config.json",
        {
            "experiment_id": "EXP-002",
            "tau": 1.0,
            "lambda_shift": 2.5,
            "embedding_dimensions": 4,
        },
    )
    write_json(
        exp002 / "metrics.json",
        {
            "candidate_pair_count": int(len(feature_frame)),
            "feature_non_null_rate": float(feature_frame.notna().mean().mean()),
            "score_min": float(feature_frame["score"].min()),
            "score_max": float(feature_frame["score"].max()),
        },
    )
    (exp002 / "run.md").write_text(
        "# EXP-002\n\nExact toy spectral/transport pair features materialized for all drug-disease candidates.\n"
    )
    print(f"Wrote {exp001} and {exp002}")


if __name__ == "__main__":
    main()
