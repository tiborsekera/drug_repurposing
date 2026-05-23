# Feedback Response

## Revised Core Claim

The project should not claim novelty from using spectral graph features on
PrimeKG. The sharper research question is:

> Do disorder-response fingerprints identify structurally fragile drug-disease
> predictions that conventional link-prediction confidence misses?

The main paper should test whether perturbation stability adds signal after
controlling for raw score, drug degree, disease degree, and graph proximity.

## Immediate Correctness Changes

The first implementation priority is label-leakage prevention. Validation and
test drug-disease label edges must not remain in the graph used to compute
features for those same candidate pairs.

Implemented safeguards:

- `SplitGraphBundle` explicitly separates labelled pairs from the graph used
  for scoring.
- `build_split_graph_bundle()` removes held-out validation/test positive label
  edges from `candidate_graph_for_scoring`.
- Label removal handles direct and reciprocal drug-disease orientations.
- Default label relations are `indication`, `off-label use`, and
  `contraindication`.
- `leakage_report_for_split()` emits machine-readable checks for direct label
  edges, reciprocal label edges, pair overlaps, held-out disease leakage,
  held-out drug leakage, and alias-normalization status.
- `disease_held_out_split()` now samples validation indices before resetting
  indices, avoiding train/validation contamination.
- Tests now prove that changing a held-out direct label-edge weight does not
  change spectral features computed from the sanitized scoring graph.

Alias and synonym normalization is not implemented yet. The leakage report
therefore marks alias normalization as required before benchmark claims.

## Disorder Response Fingerprint

The first implementation of the response fingerprint is in
`drug_repurposing.fingerprint`.

Current quantities:

- `mean_score`
- `score_variance`
- `score_std`
- `robust_score = mean_score - beta * score_variance`
- `degradation_slope`
- `p_top_k`
- `rank_entropy`

Planned additions:

- relation-family dropout deltas
- biological-layer dropout deltas
- degree-preserving rewire null gap
- provenance dropout sensitivity
- temporal-truncation indicators when timestamped evidence is available

## Updated Priority Backlog

P0:

1. Extend split bundles to write `leakage_checks.json` for every registered
   split experiment.
2. Add hard checks for duplicate/reciprocal label aliases after entity
   normalization.
3. Add degree distribution summaries for positives and negatives.
4. Add tests for random, disease-held-out, and drug-held-out pair overlap.

P1:

1. Full PrimeKG ingestion, not head-row smoke sampling.
2. Relation inventory for label edges, contraindications, off-label use, and
   duplicated drug-disease semantics.
3. Stable integer node IDs and artifact hashes.

P2:

1. Sparse pairwise heat-kernel approximations.
2. Sparse pairwise resolvent solves.
3. Exact-versus-approximate validation on small graph slices.

P3:

1. Degree-preserving rewiring.
2. Neighborhood masking.
3. Biological-layer and provenance dropout.

P4:

1. KG embedding baselines.
2. R-GCN or HGT baseline.
3. Stability-aware shallow reranker before any deep reranker.
