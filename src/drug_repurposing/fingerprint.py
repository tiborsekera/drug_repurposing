"""Disorder response fingerprints for candidate drug-disease pairs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {"pair_id", "perturbation_strength", "realization_id", "score"}


def add_ranks(
    score_realizations: pd.DataFrame,
    score_column: str = "score",
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Add descending ranks within each perturbation strength and realization."""

    missing = REQUIRED_SCORE_COLUMNS - set(score_realizations.columns)
    if missing:
        raise ValueError(f"Missing score realization columns: {sorted(missing)}")

    ranked = score_realizations.copy()
    ranked[rank_column] = ranked.groupby(["perturbation_strength", "realization_id"])[score_column].rank(
        method="min",
        ascending=False,
    )
    return ranked


def rank_entropy(ranks: pd.Series) -> float:
    """Compute entropy of observed ranks for a single candidate."""

    counts = ranks.value_counts(normalize=True)
    return float(-sum(probability * math.log(probability) for probability in counts if probability > 0))


def degradation_slope(strengths: pd.Series, scores: pd.Series) -> float:
    """Fit score mean as a linear function of perturbation strength."""

    x = strengths.to_numpy(dtype=float)
    y = scores.to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def disorder_response_fingerprint(
    score_realizations: pd.DataFrame,
    top_k: int = 10,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Summarize perturbation response for each candidate pair.

    Expected columns are `pair_id`, `perturbation_strength`, `realization_id`,
    and `score`. If a `rank` column is absent, ranks are derived from score.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if "rank" not in score_realizations.columns:
        score_realizations = add_ranks(score_realizations)

    rows = []
    for pair_id, pair_frame in score_realizations.groupby("pair_id", sort=True):
        by_strength = (
            pair_frame.groupby("perturbation_strength", as_index=False)["score"]
            .mean()
            .rename(columns={"score": "mean_score_at_strength"})
        )
        mean_score = float(pair_frame["score"].mean())
        score_variance = float(pair_frame["score"].var(ddof=0))
        rows.append(
            {
                "pair_id": pair_id,
                "mean_score": mean_score,
                "score_variance": score_variance,
                "score_std": float(math.sqrt(score_variance)),
                "robust_score": float(mean_score - beta * score_variance),
                "degradation_slope": degradation_slope(
                    by_strength["perturbation_strength"],
                    by_strength["mean_score_at_strength"],
                ),
                "p_top_k": float(np.mean(pair_frame["rank"].to_numpy(dtype=float) <= top_k)),
                "rank_entropy": rank_entropy(pair_frame["rank"]),
                "n_realizations": int(len(pair_frame)),
                "n_strengths": int(pair_frame["perturbation_strength"].nunique()),
            }
        )
    return pd.DataFrame(rows)
