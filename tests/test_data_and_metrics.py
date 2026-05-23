import numpy as np
import pandas as pd

from drug_repurposing.data import (
    graph_to_pair_frame,
    primekg_to_graph,
    random_pair_split,
    sample_negative_pairs,
)
from drug_repurposing.metrics import binary_ranking_metrics, expected_calibration_error, hits_at_k


def test_primekg_like_frame_converts_to_graph_and_pairs():
    frame = pd.DataFrame(
        [
            {
                "x_id": "drug_a",
                "x_type": "drug",
                "y_id": "disease_a",
                "y_type": "disease",
                "display_relation": "indication",
            },
            {
                "x_id": "gene_a",
                "x_type": "gene",
                "y_id": "disease_a",
                "y_type": "disease",
                "display_relation": "associated with",
            },
        ]
    )

    graph = primekg_to_graph(frame)
    pairs = graph_to_pair_frame(graph)

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert pairs.to_dict("records") == [{"drug_id": "drug_a", "disease_id": "disease_a", "label": 1}]


def test_negative_sampling_and_random_split_are_deterministic():
    frame = pd.DataFrame(
        [
            {"x_id": "drug_a", "x_type": "drug", "y_id": "disease_a", "y_type": "disease", "relation": "indication"},
            {"x_id": "drug_b", "x_type": "drug", "y_id": "disease_b", "y_type": "disease", "relation": "indication"},
        ]
    )
    graph = primekg_to_graph(frame)
    positives = graph_to_pair_frame(graph)
    negatives = sample_negative_pairs(graph, positives, n_negatives=2, seed=7)
    repeated = sample_negative_pairs(graph, positives, n_negatives=2, seed=7)

    assert negatives.equals(repeated)
    assert set(zip(negatives["drug_id"], negatives["disease_id"])).isdisjoint(
        set(zip(positives["drug_id"], positives["disease_id"]))
    )

    split = random_pair_split(positives, negatives, seed=11, validation_fraction=0.25, test_fraction=0.25)
    assert len(split.train) + len(split.validation) + len(split.test) == 4


def test_metrics_cover_ranking_and_calibration():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.7, 0.2])

    metrics = binary_ranking_metrics(labels, scores)

    assert metrics["auprc"] == 1.0
    assert metrics["auroc"] == 1.0
    assert hits_at_k(labels, scores, k=1) == 1.0
    assert expected_calibration_error(labels, scores, n_bins=2) >= 0.0
