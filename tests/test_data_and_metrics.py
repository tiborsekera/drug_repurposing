import numpy as np
import pandas as pd

from drug_repurposing.data import (
    PairSplit,
    build_split_graph_bundle,
    disease_held_out_split,
    graph_to_pair_frame,
    primekg_to_graph,
    random_pair_split,
    sample_negative_pairs,
)
from drug_repurposing.features import pair_features
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


def test_disease_held_out_split_does_not_drop_wrong_validation_rows():
    positives = pd.DataFrame(
        [
            {"drug_id": "drug_a", "disease_id": "disease_train_a", "label": 1},
            {"drug_id": "drug_b", "disease_id": "disease_train_b", "label": 1},
            {"drug_id": "drug_c", "disease_id": "disease_train_c", "label": 1},
            {"drug_id": "drug_d", "disease_id": "disease_held_out", "label": 1},
        ]
    )
    negatives = pd.DataFrame(
        [
            {"drug_id": "drug_e", "disease_id": "disease_train_d", "label": 0},
            {"drug_id": "drug_f", "disease_id": "disease_held_out", "label": 0},
        ]
    )

    split = disease_held_out_split(
        positives,
        negatives,
        held_out_diseases={"disease_held_out"},
        validation_fraction=0.5,
        seed=3,
    )

    train_pairs = set(zip(split.train["drug_id"], split.train["disease_id"]))
    validation_pairs = set(zip(split.validation["drug_id"], split.validation["disease_id"]))
    test_pairs = set(zip(split.test["drug_id"], split.test["disease_id"]))

    assert train_pairs.isdisjoint(validation_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert validation_pairs.isdisjoint(test_pairs)
    assert all(disease != "disease_held_out" for disease in split.train["disease_id"])
    assert all(disease == "disease_held_out" for disease in split.test["disease_id"])


def test_split_graph_bundle_removes_held_out_label_edges_and_reports_leakage():
    frame = pd.DataFrame(
        [
            {"x_id": "drug_a", "x_type": "drug", "y_id": "disease_a", "y_type": "disease", "relation": "indication"},
            {"x_id": "disease_b", "x_type": "disease", "y_id": "drug_b", "y_type": "drug", "relation": "indication"},
            {"x_id": "drug_a", "x_type": "drug", "y_id": "gene_x", "y_type": "gene", "relation": "targets"},
            {"x_id": "gene_x", "x_type": "gene", "y_id": "disease_b", "y_type": "disease", "relation": "associated_with"},
        ]
    )
    graph = primekg_to_graph(frame)
    split = PairSplit(
        split_id="unit",
        train=pd.DataFrame([{"drug_id": "drug_a", "disease_id": "disease_a", "label": 1}]),
        validation=pd.DataFrame(columns=["drug_id", "disease_id", "label"]),
        test=pd.DataFrame([{"drug_id": "drug_b", "disease_id": "disease_b", "label": 1}]),
    )

    bundle = build_split_graph_bundle(graph, split)

    train_edges = {(edge.source, edge.relation, edge.target) for edge in bundle.train_graph.edges}
    assert ("disease_b", "indication", "drug_b") not in train_edges
    assert ("drug_a", "indication", "disease_a") in train_edges
    assert bundle.removed_label_edges.to_dict("records") == [
        {
            "source": "disease_b",
            "target": "drug_b",
            "relation": "indication",
            "weight": 1.0,
            "drug_id": "drug_b",
            "disease_id": "disease_b",
        }
    ]
    assert bundle.leakage_report["direct_label_edges_absent"]
    assert bundle.leakage_report["reciprocal_label_edges_absent"]
    assert bundle.leakage_report["pair_overlap_zero"]


def test_held_out_label_edge_cannot_influence_spectral_features_after_sanitization():
    def make_graph(held_out_weight):
        return primekg_to_graph(
            pd.DataFrame(
                [
                    {
                        "x_id": "drug_a",
                        "x_type": "drug",
                        "y_id": "disease_a",
                        "y_type": "disease",
                        "relation": "indication",
                    },
                    {
                        "x_id": "drug_b",
                        "x_type": "drug",
                        "y_id": "disease_b",
                        "y_type": "disease",
                        "relation": "indication",
                    },
                    {
                        "x_id": "drug_b",
                        "x_type": "drug",
                        "y_id": "gene_x",
                        "y_type": "gene",
                        "relation": "targets",
                    },
                    {
                        "x_id": "gene_x",
                        "x_type": "gene",
                        "y_id": "disease_b",
                        "y_type": "disease",
                        "relation": "associated_with",
                    },
                ]
            )
        ).without_edges(
            [
                edge
                if not (edge.source == "drug_b" and edge.target == "disease_b")
                else type(edge)(edge.source, edge.target, edge.relation, held_out_weight)
                for edge in primekg_to_graph(
                    pd.DataFrame(
                        [
                            {
                                "x_id": "drug_a",
                                "x_type": "drug",
                                "y_id": "disease_a",
                                "y_type": "disease",
                                "relation": "indication",
                            },
                            {
                                "x_id": "drug_b",
                                "x_type": "drug",
                                "y_id": "disease_b",
                                "y_type": "disease",
                                "relation": "indication",
                            },
                            {
                                "x_id": "drug_b",
                                "x_type": "drug",
                                "y_id": "gene_x",
                                "y_type": "gene",
                                "relation": "targets",
                            },
                            {
                                "x_id": "gene_x",
                                "x_type": "gene",
                                "y_id": "disease_b",
                                "y_type": "disease",
                                "relation": "associated_with",
                            },
                        ]
                    )
                ).edges
            ]
        )

    split = PairSplit(
        split_id="unit",
        train=pd.DataFrame([{"drug_id": "drug_a", "disease_id": "disease_a", "label": 1}]),
        validation=pd.DataFrame(columns=["drug_id", "disease_id", "label"]),
        test=pd.DataFrame([{"drug_id": "drug_b", "disease_id": "disease_b", "label": 1}]),
    )
    bundle_low = build_split_graph_bundle(make_graph(1.0), split)
    bundle_high = build_split_graph_bundle(make_graph(999.0), split)

    features_low = pair_features(bundle_low.candidate_graph_for_scoring, "drug_b", "disease_b").as_array()
    features_high = pair_features(bundle_high.candidate_graph_for_scoring, "drug_b", "disease_b").as_array()

    assert np.allclose(features_low, features_high)
