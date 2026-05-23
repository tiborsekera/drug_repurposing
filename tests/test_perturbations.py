import numpy as np
import pytest

from drug_repurposing.perturbations import edge_dropout, summarize_score_robustness


def toy_heterogeneous_edges():
    return [
        ("drug_a", "targets", "protein_x", 1.0),
        ("drug_b", "targets", "protein_y", 1.0),
        ("protein_x", "interacts", "protein_y", 0.5),
        ("protein_x", "participates_in", "pathway_m", 1.0),
        ("protein_y", "participates_in", "pathway_m", 1.0),
        ("pathway_m", "associated_with", "disease_alpha", 1.0),
        ("protein_y", "associated_with", "disease_beta", 0.75),
    ]


def toy_pair_scores(edges):
    edge_set = {(source, relation, target) for source, relation, target, _ in edges}
    return {
        ("drug_a", "disease_alpha"): float(
            ("drug_a", "targets", "protein_x") in edge_set
            and ("protein_x", "participates_in", "pathway_m") in edge_set
            and ("pathway_m", "associated_with", "disease_alpha") in edge_set
        ),
        ("drug_b", "disease_beta"): float(
            ("drug_b", "targets", "protein_y") in edge_set
            and ("protein_y", "associated_with", "disease_beta") in edge_set
        ),
    }


def test_edge_dropout_is_deterministic_with_seed_and_preserves_schema():
    edges = toy_heterogeneous_edges()

    first = edge_dropout(edges, drop_probability=0.35, seed=17)
    second = edge_dropout(edges, drop_probability=0.35, seed=17)
    different_seed = edge_dropout(edges, drop_probability=0.35, seed=23)

    assert first == second
    assert first != different_seed
    assert all(len(edge) == 4 for edge in first)
    assert all(edge in edges for edge in first)
    assert 0 < len(first) < len(edges)


def test_score_robustness_summary_fields_are_complete_and_consistent():
    edges = toy_heterogeneous_edges()
    pairs = [("drug_a", "disease_alpha"), ("drug_b", "disease_beta")]

    def score_fn(perturbed_edges):
        return toy_pair_scores(perturbed_edges)

    summary = summarize_score_robustness(
        edges=edges,
        pairs=pairs,
        score_fn=score_fn,
        perturbation=edge_dropout,
        perturbation_kwargs={"drop_probability": 0.25},
        seeds=[3, 5, 7, 11, 13],
        beta=0.5,
    )

    expected_fields = {
        "pair",
        "mean_score",
        "score_variance",
        "score_std",
        "min_score",
        "max_score",
        "robust_score",
        "n_perturbations",
    }

    assert len(summary) == 2
    assert {row["pair"] for row in summary} == set(pairs)

    for row in summary:
        assert expected_fields <= row.keys()
        assert row["n_perturbations"] == 5
        assert 0.0 <= row["min_score"] <= row["mean_score"] <= row["max_score"] <= 1.0
        assert row["score_variance"] >= 0.0
        assert row["score_std"] == pytest.approx(np.sqrt(row["score_variance"]))
        assert row["robust_score"] == pytest.approx(
            row["mean_score"] - 0.5 * row["score_variance"]
        )
