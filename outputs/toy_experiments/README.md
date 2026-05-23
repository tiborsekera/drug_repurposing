# Toy experiment outputs

`scripts/run_toy_experiments.py` writes deterministic outputs for the built-in
toy biomedical graph. It scores every drug-disease candidate pair with spectral
features, then runs edge-dropout robustness sweeps with `indication` edges
protected.

Run from the repository root:

```bash
python3 scripts/run_toy_experiments.py
```

The default output directory is `outputs/toy_experiments/`.

## `features.csv`

One row per candidate drug-disease pair.

| Column | Description |
| --- | --- |
| `pair_id` | Stable identifier derived from source and target IDs. |
| `source` | Drug node ID. |
| `target` | Disease node ID. |
| `source_type` | Source node type, expected to be `drug`. |
| `target_type` | Target node type, expected to be `disease`. |
| `is_known_indication` | Whether the toy graph has an `indication` edge for the pair. |
| `heat` | Heat-kernel pair entry from `drug_repurposing.features.pair_features`. |
| `resolvent` | Resolvent pair entry from `pair_features`. |
| `low_frequency_distance` | Euclidean distance in the low-frequency Laplacian embedding. |
| `source_degree` | Weighted source degree in the toy graph adjacency. |
| `target_degree` | Weighted target degree in the toy graph adjacency. |
| `nominal_score` | `heat + 0.25 * resolvent - low_frequency_distance`. |

## `robustness.csv`

One row per candidate pair and dropout probability.

| Column | Description |
| --- | --- |
| `pair_id` | Stable candidate identifier matching `features.csv`. |
| `source` | Drug node ID. |
| `target` | Disease node ID. |
| `dropout_probability` | Edge removal probability for non-protected edges. |
| `protected_relations` | Pipe-delimited relation names preserved during dropout. |
| `beta` | Variance penalty used in robust scoring. |
| `seed` | Deterministic sweep seed derived from the base seed, pair index, and dropout index. |
| `n_realizations` | Number of perturbation realizations in the row summary. |
| `mean_score` | Mean score across perturbed graphs. |
| `variance` | Population variance of scores across perturbed graphs. |
| `robust_score` | `mean_score - beta * variance`. |
| `min_score` | Minimum score across perturbation realizations. |
| `max_score` | Maximum score across perturbation realizations. |

## `metrics.json`

Experiment metadata and compact summary values:

- `experiment`, `seed`, `beta`, `n_realizations`, and
  `dropout_probabilities` describe the run configuration.
- `protected_relations` lists relations exempt from dropout.
- `graph` records toy graph size and relation names.
- `n_candidate_pairs`, `n_feature_rows`, and `n_robustness_rows` are row-count
  checks.
- `score_formula` records the scoring rule used by both CSV outputs.
- `top_nominal_pair` gives the highest-scoring pair from `features.csv`.
- `top_robust_pair_by_dropout` gives the highest `robust_score` row for each
  dropout probability.
