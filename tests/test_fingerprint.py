import pandas as pd
import pytest

from drug_repurposing.fingerprint import add_ranks, disorder_response_fingerprint


def test_add_ranks_ranks_within_strength_and_realization():
    realizations = pd.DataFrame(
        [
            {"pair_id": "a", "perturbation_strength": 0.0, "realization_id": 0, "score": 0.9},
            {"pair_id": "b", "perturbation_strength": 0.0, "realization_id": 0, "score": 0.4},
            {"pair_id": "a", "perturbation_strength": 0.5, "realization_id": 0, "score": 0.2},
            {"pair_id": "b", "perturbation_strength": 0.5, "realization_id": 0, "score": 0.8},
        ]
    )

    ranked = add_ranks(realizations)

    assert ranked.loc[ranked["pair_id"] == "a", "rank"].tolist() == [1.0, 2.0]
    assert ranked.loc[ranked["pair_id"] == "b", "rank"].tolist() == [2.0, 1.0]


def test_disorder_response_fingerprint_summarizes_stability():
    realizations = pd.DataFrame(
        [
            {"pair_id": "stable", "perturbation_strength": 0.0, "realization_id": 0, "score": 1.0},
            {"pair_id": "fragile", "perturbation_strength": 0.0, "realization_id": 0, "score": 1.0},
            {"pair_id": "stable", "perturbation_strength": 0.5, "realization_id": 0, "score": 0.9},
            {"pair_id": "fragile", "perturbation_strength": 0.5, "realization_id": 0, "score": 0.2},
            {"pair_id": "stable", "perturbation_strength": 1.0, "realization_id": 0, "score": 0.8},
            {"pair_id": "fragile", "perturbation_strength": 1.0, "realization_id": 0, "score": 0.0},
        ]
    )

    fingerprint = disorder_response_fingerprint(realizations, top_k=1, beta=1.0).set_index("pair_id")

    assert fingerprint.loc["stable", "score_variance"] < fingerprint.loc["fragile", "score_variance"]
    assert fingerprint.loc["stable", "degradation_slope"] == pytest.approx(-0.2)
    assert fingerprint.loc["fragile", "degradation_slope"] == pytest.approx(-1.0)
    assert fingerprint.loc["stable", "p_top_k"] > fingerprint.loc["fragile", "p_top_k"]
    assert fingerprint.loc["stable", "rank_entropy"] >= 0.0
