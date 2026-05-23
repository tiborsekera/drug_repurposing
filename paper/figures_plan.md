# Figure Plan

This plan maps each manuscript figure to concrete experiment IDs and required artifacts. Figure panels should not be generated from live training code; they should read from the stored outputs listed here.

## Figure 1: Disorder-aware drug-repurposing workflow

Manuscript cross-reference: Result 1.

Purpose: Introduce the conceptual pipeline from biomedical knowledge graph to sparse operator, pair-level spectral/transport features, perturbation ensemble, robust score, and ranked drug-disease candidates.

Panels:

- 1a: Heterogeneous biomedical KG schematic with drug, disease, gene, pathway, side-effect, and phenotype node types.
- 1b: Operator view showing relation-weighted adjacency and normalized Laplacian construction.
- 1c: Candidate pair feature extraction: heat-kernel score, resolvent score, low-frequency distance, and degree covariates.
- 1d: Disorder ensemble producing score mean, score variance, and robust score.

Required artifacts:

- `artifacts/EXP-001/entity_table.parquet`
- `artifacts/EXP-001/relation_table.parquet`
- `artifacts/EXP-001/edge_table.parquet`
- `artifacts/EXP-002/pair_features.parquet`
- `artifacts/EXP-002/metrics.json`

Status: Partially complete. `EXP-001`, `EXP-002`, and `figures/data/toy_feature_metrics.json` now exist for toy/API panels. Final plotted panels are still TODO.

## Figure 2: Graph operators and relation-channel structure

Manuscript cross-reference: Results 1-3 and Methods.

Purpose: Show how heterogeneous relation channels are converted into sparse operators and verify graph statistics before predictive claims.

Panels:

- 2a: Toy graph relation-channel matrices for the implemented MVP.
- 2b: Spectrum of the toy normalized Laplacian with the low-frequency modes used for pair features.
- 2c: PrimeKG relation count and node-type count summary after ingestion.
- 2d: PrimeKG connected component and drug/disease degree summaries.

Required artifacts:

- `artifacts/EXP-001/adjacency.npz`
- `artifacts/EXP-002/pair_features.parquet`
- `artifacts/EXP-010/entities.parquet`
- `artifacts/EXP-010/relations.parquet`
- `artifacts/EXP-010/edges.parquet`
- `artifacts/EXP-010/metrics.json`

Status: Partially complete. `figures/data/toy_laplacian_spectrum.csv` exists for panel 2b. A PrimeKG 50,000-row smoke metric file exists under `outputs/primekg_smoke/metrics.json`, but it is not a substitute for full `EXP-010` because the head sample contains only PPI rows.

## Figure 3: Predictive performance across split regimes

Manuscript cross-reference: Result 4.

Purpose: Compare degree/network baselines, embedding baselines, heterogeneous GNN baseline if run, spectral-only models, and hybrid models across random, disease-held-out, drug-held-out, and disease-class-held-out splits.

Panels:

- 3a: AUPRC by model and split.
- 3b: Hits@K or Recall@K by model and split.
- 3c: Expected calibration error by model and split.
- 3d: Metric deltas for spectral-only and hybrid models relative to the selected baseline.

Required artifacts:

- `artifacts/EXP-020/train.parquet`, `validation.parquet`, `test.parquet`, `leakage_checks.json`
- `artifacts/EXP-021/train.parquet`, `validation.parquet`, `test.parquet`, `leakage_checks.json`
- `artifacts/EXP-022/train.parquet`, `validation.parquet`, `test.parquet`, `leakage_checks.json`
- `artifacts/EXP-023/train.parquet`, `validation.parquet`, `test.parquet`, `leakage_checks.json`
- `artifacts/EXP-030/predictions.parquet`, `metrics.json`
- `artifacts/EXP-031/predictions.parquet`, `metrics.json`
- `artifacts/EXP-032/predictions.parquet`, `metrics.json` if the GNN baseline is feasible
- `artifacts/EXP-050/predictions.parquet`, `metrics.json`
- `artifacts/EXP-051/predictions.parquet`, `metrics.json`
- `artifacts/EXP-060/model_comparison_table.parquet`

Status: TODO. No benchmark results are currently available.

## Figure 4: Robustness phase diagram under graph disorder

Manuscript cross-reference: Result 5.

Purpose: Test whether model performance degrades more slowly for robust predictions as perturbation strength increases.

Panels:

- 4a: AUPRC versus edge-dropout strength.
- 4b: AUPRC or Hits@K versus weight-noise strength.
- 4c: Relation-dropout sensitivity heatmap by relation family and model.
- 4d: Robustness area or degradation slope by model and split.

Required artifacts:

- `artifacts/EXP-040/score_realizations.parquet`, `predictions.parquet`, `metrics.json`
- `artifacts/EXP-041/score_realizations.parquet`, `predictions.parquet`, `metrics.json`
- `artifacts/EXP-042/score_realizations.parquet`, `predictions.parquet`, `metrics.json`
- `artifacts/EXP-043/score_realizations.parquet`, `predictions.parquet`, `metrics.json`
- `artifacts/EXP-044/score_realizations.parquet`, `predictions.parquet`, `metrics.json`
- `artifacts/EXP-045/score_realizations.parquet`, `predictions.parquet`, `metrics.json`
- `artifacts/EXP-052/reranked_candidates.parquet`
- `artifacts/EXP-061/robustness_phase_diagram_data.parquet`

Status: TODO. Current code supports edge dropout, relation dropout, weight noise, and score aggregation, but the full perturbation sweep has not been run.

## Figure 5: Score stability separates robust and fragile predictions

Manuscript cross-reference: Result 6.

Purpose: Determine whether perturbation variance distinguishes true positives from false positives and improves calibration.

Panels:

- 5a: Score-variance distributions for true positives, false positives, true negatives, and false negatives.
- 5b: Calibration curves before and after robust reranking.
- 5c: Scatter plot of baseline score versus score variance, colored by outcome.
- 5d: AUROC or AUPRC using stability-derived features alone.

Required artifacts:

- `artifacts/EXP-040/score_realizations.parquet`
- `artifacts/EXP-052/reranked_candidates.parquet`
- `artifacts/EXP-062/variance_by_outcome.parquet`
- `artifacts/EXP-062/calibration_bins.parquet`
- `artifacts/EXP-062/metrics.json`

Status: TODO. No stability/outcome analysis has been run.

## Figure 6: Case studies of robust and fragile drug-disease candidates

Manuscript cross-reference: Result 6 and Discussion.

Purpose: Show how relation-channel sensitivity and path recurrence support cautious interpretation of selected candidate predictions.

Panels:

- 6a: Table of selected robust and fragile candidate pairs with baseline rank, robust rank, score mean, and score variance.
- 6b: Relation-channel sensitivity profile for each selected pair.
- 6c: Recurrent paths across perturbation realizations for robust candidates.
- 6d: Fragile high-score example whose support collapses under relation or edge perturbation.

Required artifacts:

- `artifacts/EXP-041/predictions.parquet`
- `artifacts/EXP-045/predictions.parquet`
- `artifacts/EXP-052/reranked_candidates.parquet`
- `artifacts/EXP-063/case_selection_config.yaml`
- `artifacts/EXP-063/case_studies.parquet`
- `artifacts/EXP-063/path_recurrence.parquet`

Status: TODO. Case studies must be selected by a pre-specified rule and should not be framed as validated therapeutic discoveries.

## Extended Data Figure 1: Exact versus approximate spectral features

Manuscript cross-reference: Result 2.

Purpose: Validate sparse approximations before scaling beyond the toy graph.

Panels:

- ED1a: Exact versus approximate heat-kernel entries.
- ED1b: Exact versus approximate resolvent entries.
- ED1c: Rank correlation for candidate-pair scores.
- ED1d: Runtime and memory by graph size.

Required artifacts:

- `artifacts/EXP-003/pair_features_exact.parquet`
- `artifacts/EXP-003/pair_features_approx.parquet`
- `artifacts/EXP-003/approximation_error.parquet`
- `artifacts/EXP-003/metrics.json`

Status: TODO.

## Extended Data Figure 2: Split diagnostics and leakage checks

Manuscript cross-reference: Result 3.

Purpose: Document that the benchmark splits are valid and materially different in difficulty.

Panels:

- ED2a: Positive and negative counts by split.
- ED2b: Degree distributions for positives and sampled negatives.
- ED2c: Held-out disease and held-out drug leakage checks.
- ED2d: Disease-class-held-out class separation summary.

Required artifacts:

- `artifacts/EXP-020/metrics.json`, `leakage_checks.json`
- `artifacts/EXP-021/metrics.json`, `leakage_checks.json`
- `artifacts/EXP-022/metrics.json`, `leakage_checks.json`
- `artifacts/EXP-023/metrics.json`, `leakage_checks.json`

Status: TODO.

## Extended Data Figure 3: Baseline and spectral feature ablations

Manuscript cross-reference: Result 4.

Purpose: Show which feature families contribute to prediction, calibration, or robustness.

Panels:

- ED3a: Spectral-only feature importances or coefficients.
- ED3b: Hybrid model ablation excluding heat, resolvent, low-frequency distance, or degree features.
- ED3c: Degree-stratified AUPRC to assess degree bias.
- ED3d: Negative-sampling sensitivity analysis.

Required artifacts:

- `artifacts/EXP-030/metrics.json`
- `artifacts/EXP-050/predictions.parquet`, `metrics.json`
- `artifacts/EXP-051/predictions.parquet`, `metrics.json`
- Additional ablation outputs under `artifacts/EXP-051/ablations/`

Status: TODO.

## Extended Data Figure 4: Perturbation-family diagnostics

Manuscript cross-reference: Result 5.

Purpose: Confirm that each perturbation family changes graph structure in the intended way.

Panels:

- ED4a: Edge retention and component statistics by edge-dropout strength.
- ED4b: Weight distribution before and after weight-noise perturbation.
- ED4c: Degree distribution preservation under degree-preserving rewiring.
- ED4d: Relation or biological-layer coverage after dropout.

Required artifacts:

- `artifacts/EXP-040/metrics.json`
- `artifacts/EXP-042/metrics.json`
- `artifacts/EXP-044/metrics.json`
- `artifacts/EXP-045/metrics.json`

Status: TODO.

## Extended Data Figure 5: Disorder-aware reranking behavior

Manuscript cross-reference: Result 5.

Purpose: Inspect how robust reranking changes candidate order and whether it trades clean performance for stability.

Panels:

- ED5a: Baseline rank versus robust rank.
- ED5b: Distribution of rank changes by label and split.
- ED5c: Clean AUPRC versus robust AUPRC tradeoff by beta.
- ED5d: Score mean versus variance for reranked top candidates.

Required artifacts:

- `artifacts/EXP-052/reranked_candidates.parquet`
- `artifacts/EXP-052/metrics.json`
- `artifacts/EXP-061/robustness_phase_diagram_data.parquet`

Status: TODO.

## Extended Data Figure 6: Case-study evidence audit

Manuscript cross-reference: Result 6.

Purpose: Prevent overinterpretation of selected candidates by documenting evidence channels, fragility, and external annotation status.

Panels:

- ED6a: Relation family contribution table for selected cases.
- ED6b: Path recurrence and path collapse under dropout.
- ED6c: External annotation fields and whether the pair is known, ambiguous, or unsupported.
- ED6d: Sensitivity of case selection to robust-score beta.

Required artifacts:

- `artifacts/EXP-063/case_studies.parquet`
- `artifacts/EXP-063/path_recurrence.parquet`
- `artifacts/EXP-063/case_selection_config.yaml`
- Optional external annotation table, if available.

Status: TODO.
