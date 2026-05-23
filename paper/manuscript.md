# Disorder-aware spectral graph learning for robust drug repurposing

## Abstract

Biomedical knowledge graphs provide a scalable substrate for drug repurposing, but predictions derived from these graphs can depend strongly on incomplete edges, uneven relation coverage, and degree-driven shortcuts. We propose a disorder-aware spectral graph learning framework that treats a biomedical knowledge graph as a sparse operator and evaluates candidate drug-disease predictions through spectral, transport-style, and perturbation-stability features. The current implementation supports a minimal heterogeneous graph API, normalized Laplacian construction, heat-kernel pair features, resolvent features, low-frequency embedding distances, edge dropout, relation dropout, weight noise, and mean-variance robustness summaries. We have materialized deterministic toy-graph artifacts for the first two registered experiments and run a bounded PrimeKG ingestion smoke check. The planned full analysis will test whether these features improve robustness, calibration, or interpretability of drug-disease link prediction on PrimeKG-derived splits, with DRKG reserved as an external replication graph if resources permit. No PrimeKG-scale predictive or robustness results have been generated yet. This manuscript is therefore a serious skeleton and analysis plan: all unrun analyses are marked as TODO with required artifacts and experiment identifiers.

## Introduction

Drug repurposing from biomedical knowledge graphs has become an attractive strategy because heterogeneous resources can connect drugs, diseases, genes, pathways, phenotypes, side effects, and clinical indications in a single predictive structure. Graph neural networks and knowledge-graph embedding models can exploit these relationships to rank candidate drug-disease links, including in zero-shot settings where diseases or drugs are held out during training. However, the same graph structure that gives these models broad reach also creates a central uncertainty: a high score may reflect robust biological support, or it may reflect local graph density, missing evidence, idiosyncratic relation coverage, or reliance on a single fragile path.

The project is motivated by the hypothesis that confidence without perturbation stability is incomplete. A drug-disease prediction should not only rank highly on the observed graph; it should remain comparatively stable when the graph is subjected to perturbations that mimic missing literature evidence, uncertain edge weights, relation-specific bias, or sparse drug and disease neighborhoods. This framing shifts the contribution away from proposing another drug-repurposing model and toward measuring when graph-derived predictions are stable, fragile, or explainable.

We define sparse graph operators over a heterogeneous biomedical knowledge graph and compute pair-level features for candidate drug-disease links. The implemented MVP includes a weighted adjacency matrix, the symmetric normalized Laplacian, heat-kernel scores, resolvent scores, low-frequency embedding distances, and simple degree covariates. A disorder ensemble then repeatedly perturbs the graph and summarizes each candidate score by its mean, variance, and variance-penalized robust score. At full scale, these quantities will be combined with network proximity, knowledge-graph embeddings, and optional heterogeneous GNN baselines under random, disease-held-out, drug-held-out, and disease-class-held-out splits.

This manuscript skeleton is deliberately conservative. It does not report biological discoveries, benchmark improvements, or robustness effects that have not yet been observed. Instead, it specifies the analyses required to support or reject the central claim that disorder-response fingerprints identify structurally fragile drug-disease predictions that conventional link-prediction confidence misses.

## Results

### Result 1: A minimal spectral-disorder API supports deterministic toy-graph experiments

The current codebase implements the core objects and operations needed for a toy-scale proof of concept. `drug_repurposing.graph` defines typed `Node`, weighted typed `Edge`, and `HeterogeneousGraph` containers, with relation-weighted adjacency construction and symmetric normalized Laplacian construction. `drug_repurposing.features` implements dense small-graph heat kernels, resolvent matrices, low-frequency Laplacian embeddings, and the `PairFeatures` record for candidate pairs. `drug_repurposing.perturbations` implements edge dropout, relation dropout, multiplicative log-normal weight noise, and robustness summaries over perturbation realizations.

Current evidence is limited to implementation, unit tests, deterministic toy artifacts, and smoke analyses. `EXP-001` materializes a seven-node, eight-edge heterogeneous toy graph with four relation types and a sparse adjacency. `EXP-002` materializes exact spectral/transport features for all four toy drug-disease candidates. The deterministic toy experiment runner additionally writes `outputs/toy_experiments/features.csv`, `robustness.csv`, and `metrics.json`; these outputs test the mechanics of edge-dropout robustness summaries but are not biological evidence.

Experiment IDs: `EXP-001`, `EXP-002`.

Figure cross-reference: Figure 1 and Figure 2 in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

Completed artifacts:

- `artifacts/EXP-001/config.json`, `entity_table.csv`, `relation_table.csv`, `edge_table.csv`, `adjacency.npz`, `metrics.json`, and `run.md`.
- `artifacts/EXP-002/config.json`, `pair_features.csv`, `metrics.json`, and `run.md`.
- `figures/data/toy_laplacian_spectrum.csv` and `figures/data/toy_feature_metrics.json`.
- `outputs/toy_experiments/features.csv`, `robustness.csv`, and `metrics.json`.

Remaining hardening:

- Add immutable checksums to `EXP-001` and `EXP-002`.
- Replace CSV with parquet once optional parquet dependencies are added or standardize CSV as the project format.
- Add connected-component diagnostics and runtime metadata.

### Result 2: Sparse approximations must be validated before PrimeKG-scale feature claims

The implemented feature routines use dense operations for small graphs. Full PrimeKG-scale analyses will require sparse approximations for heat-kernel entries, resolvent-like scores, and low-frequency embeddings. The first methodological result should therefore establish approximation error and runtime on toy and small graph slices before reporting any biological or benchmark outcome.

Experiment IDs: `EXP-003`, `EXP-004`, `EXP-011`.

Figure cross-reference: Figure 2 and Extended Data Figure 1 in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

TODO artifacts:

- `artifacts/EXP-003/config.yaml` specifying exact and approximate solvers, tolerances, seeds, and graph inputs.
- `artifacts/EXP-003/pair_features_exact.parquet` and `pair_features_approx.parquet`.
- `artifacts/EXP-003/approximation_error.parquet` with per-feature absolute error, relative error, and rank differences.
- `artifacts/EXP-003/metrics.json` with MAE, RMSE, Spearman rank correlation, wall-clock runtime, and peak memory if measured.
- `artifacts/EXP-004/predictions.parquet` using the common prediction schema from the experiment registry.
- `artifacts/EXP-004/metrics.json` with AUROC, AUPRC, Hits@K, MRR, and calibration error for the small KG spectral-only predictor.
- `artifacts/EXP-011/graph_slice_stats.json` with relation coverage, retained drug-disease positives, and degree distribution comparison to the parent graph.

### Result 3: PrimeKG ingestion and leakage-controlled splits define the benchmark surface

The primary planned graph is PrimeKG. The manuscript should not make claims about drug repurposing until ingestion, graph statistics, and split leakage checks are reproducible. A bounded smoke analysis successfully read 50,000 rows from the public PrimeKG Dataverse access endpoint after adding a user-agent to the loader. The first 50,000 rows were all gene/protein PPI rows and contained zero extracted drug-disease positives, which confirms that head-row smoke tests validate schema and access only; drug-disease benchmarking requires either full-file ingestion or stratified relation-aware extraction. Four split types are planned: random edge split for sanity checking, disease-held-out split for zero-shot disease generalization, drug-held-out split for new-drug generalization, and disease-class-held-out split for the hardest out-of-distribution setting.

Following internal review, leakage control is treated as a correctness blocker rather than a convenience. The implemented `SplitGraphBundle` now separates labelled pair tables from the `candidate_graph_for_scoring`. Held-out validation and test positive label edges are removed from this graph for direct and reciprocal orientations across indication, off-label-use, and contraindication relations. Machine-readable leakage reports check direct label-edge leakage, reciprocal leakage, train/validation/test pair overlap, held-out disease leakage, held-out drug leakage, and the current status of alias normalization. Alias normalization is not yet implemented, so no benchmark claim should be made until synonym and duplicate semantics are handled after entity normalization.

Experiment IDs: `EXP-010`, `EXP-020`, `EXP-021`, `EXP-022`, `EXP-023`.

Figure cross-reference: Figure 2 and Extended Data Figure 2 in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

TODO artifacts:

- `outputs/primekg_smoke/metrics.json`, `sample_edges.csv`, and `positive_pairs.csv` already exist for the bounded 50,000-row smoke check.
- `artifacts/EXP-010/config.yaml` with PrimeKG version, download/source paths, normalization rules, and relation inclusion criteria.
- `artifacts/EXP-010/inputs.json` with raw file paths and hashes.
- `artifacts/EXP-010/entities.parquet`, `relations.parquet`, `edges.parquet`, `collapsed_adjacency.npz`, and relation-specific sparse matrices.
- `artifacts/EXP-010/metrics.json` with entity counts, relation counts, duplicate rate, isolated drug/disease count, connected component statistics, and degree distribution summaries.
- `artifacts/EXP-020` through `artifacts/EXP-023` split directories, each containing `config.yaml`, `train.parquet`, `validation.parquet`, `test.parquet`, `negatives.parquet`, `leakage_checks.json`, `metrics.json`, and `run.md`.
- `leakage_checks.json` must report overlap of held-out drug or disease nodes with train positives, relation leakage through duplicate or reciprocal edges, and degree distributions of positives and sampled negatives.

### Result 4: Baselines establish whether spectral features add predictive or calibration signal

The planned model ladder begins with degree and network proximity baselines, then knowledge-graph embedding baselines, and a heterogeneous GNN baseline if feasible. Spectral-only and hybrid models will be evaluated against the same split files. The central predictive question is not whether the proposed model dominates every baseline, but whether spectral and transport features add signal under held-out disease and held-out drug conditions, especially for calibration and robust ranking.

Experiment IDs: `EXP-030`, `EXP-031`, `EXP-032`, `EXP-050`, `EXP-051`, `EXP-060`.

Figure cross-reference: Figure 3 and Extended Data Figure 3 in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

TODO artifacts:

- For each of `EXP-030`, `EXP-031`, `EXP-032`, `EXP-050`, and `EXP-051`: `config.yaml`, `inputs.json`, `predictions.parquet`, `metrics.json`, `model_uri` or `model_metadata.json`, and `run.md`.
- `predictions.parquet` must include `experiment_id`, `split_id`, `seed`, `drug_id`, `disease_id`, `label`, `score`, `rank`, `score_mean`, `score_var`, `features_uri`, and `model_uri`.
- `artifacts/EXP-060/model_comparison_table.parquet` with split-wise metrics, confidence intervals, and deltas relative to the relevant baseline.
- `artifacts/EXP-060/metrics.json` with the selected primary endpoint, secondary endpoints, and the rule used to select any highlighted model.

### Result 5: Disorder sweeps test whether robust predictions degrade more slowly

The distinct planned contribution is a disorder ensemble over graph perturbations. The perturbation families include edge dropout, relation dropout, weight noise, drug/disease neighborhood masking, degree-preserving rewiring, and biological-layer dropout. For each candidate pair and model, the analysis must store per-realization scores, not only aggregate metrics. The core test is whether stable predictions retain AUPRC or Hits@K under increasing perturbation strength and whether the degradation slope differs between baselines and disorder-aware reranking.

The first implemented fingerprint summarizes score-realization tables with mean score, score variance, variance-penalized robust score, degradation slope, probability of remaining in the top-K candidates, and rank entropy. The full manuscript should extend this to relation-family deltas, biological-layer deltas, provenance sensitivity, and a degree-preserving rewiring null gap.

Experiment IDs: `EXP-040`, `EXP-041`, `EXP-042`, `EXP-043`, `EXP-044`, `EXP-045`, `EXP-052`, `EXP-061`.

Figure cross-reference: Figure 4 and Extended Data Figures 4-5 in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

TODO artifacts:

- For each perturbation experiment `EXP-040` through `EXP-045`: `config.yaml` with perturbation family, strengths, protected relations if any, seeds, and model inputs.
- `score_realizations.parquet` with one row per `experiment_id`, `split_id`, `seed`, `perturbation_type`, `perturbation_strength`, `realization_id`, `drug_id`, `disease_id`, `label`, and `score`.
- `predictions.parquet` with score mean, score variance, robust score, and rank after aggregation.
- `metrics.json` with robust AUPRC, degradation slope, Hits@K under perturbation, calibration error, and score-variance summaries.
- `artifacts/EXP-052/reranked_candidates.parquet` with raw score, score mean, score variance, robust score, and baseline rank versus reranked rank.
- `artifacts/EXP-061/robustness_phase_diagram_data.parquet` with AUPRC and Hits@K by perturbation type, strength, model, split, and seed.

### Result 6: Stability analyses and case studies will define interpretability claims

Interpretability claims require evidence that stability summaries and relation-channel sensitivity distinguish robust from fragile predictions. Planned analyses include true-positive versus false-positive score-variance distributions, calibration bins stratified by perturbation variance, relation-channel ablations, and path recurrence summaries for selected predictions. Case studies should be selected by a pre-specified rule, such as high baseline score with low variance, high baseline score with high variance, and rank changes after robust reranking. They should not be presented as validated therapeutic discoveries without external evidence.

Experiment IDs: `EXP-041`, `EXP-045`, `EXP-062`, `EXP-063`.

Figure cross-reference: Figure 5, Figure 6, and Extended Data Figure 6 in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

TODO artifacts:

- `artifacts/EXP-062/variance_by_outcome.parquet` with labels or outcome categories, score variance, score mean, baseline score, and robust score.
- `artifacts/EXP-062/calibration_bins.parquet` with predicted probability bins, observed positive fraction, mean variance, and confidence intervals.
- `artifacts/EXP-062/metrics.json` with TP/FP variance ratio, AUROC using stability alone, ECE before and after robust reranking, and statistical test details.
- `artifacts/EXP-063/case_selection_config.yaml` with the pre-specified selection rule.
- `artifacts/EXP-063/case_studies.parquet` with drug, disease, baseline rank, robust rank, score mean, score variance, relation sensitivities, and external annotation fields.
- `artifacts/EXP-063/path_recurrence.parquet` with path identifiers, relation sequences, recurrence frequency across perturbations, and whether the path uses relation families removed in ablations.

## Discussion

The intended contribution is a robustness and interpretability layer for knowledge-graph drug repurposing. If the planned experiments succeed, the manuscript can argue that perturbation stability provides information that raw graph scores and conventional confidence estimates miss. If the spectral features do not improve headline ranking metrics, the manuscript can still be valuable if they improve calibration, identify fragile predictions, or provide interpretable relation-channel dependencies under held-out disease or drug settings. If neither predictive nor robustness analyses support the hypothesis, the paper should report that negative result and narrow the claim to the reproducible evaluation framework.

The analysis deliberately separates methodological support from biological validation. PrimeKG and related biomedical knowledge graphs encode literature and database associations; they do not by themselves establish causal therapeutic efficacy. Candidate links surfaced by this framework should be treated as hypotheses requiring independent evidence and, ultimately, experimental or clinical validation.

## Methods

### Graph representation

The implemented API represents a biomedical knowledge graph as typed nodes and typed weighted edges. `HeterogeneousGraph.adjacency()` constructs a sparse weighted adjacency matrix, optionally using relation weights and relation filters. For undirected experiments, each non-self edge is symmetrized. `HeterogeneousGraph.normalized_laplacian()` constructs the symmetric normalized Laplacian \(L = I - D^{-1/2} A D^{-1/2}\).

For tuple-based experiments, `drug_repurposing.graph_features.build_weighted_adjacency()` accepts `(source, relation, target, weight)` records and returns a sparse adjacency matrix over a fixed or inferred node order. `drug_repurposing.graph_features.normalized_laplacian()` exposes the same Laplacian construction for sparse matrices.

### Spectral and transport features

For a candidate pair `(source, target)`, the current MVP computes:

- Heat-kernel entry from `heat_kernel_matrix(laplacian, tau)`.
- Resolvent entry from `resolvent_matrix(operator, lambda_shift)`, corresponding to \((\lambda I - H)^{-1}\) for small graphs.
- Low-frequency embedding distance from the first non-trivial normalized-Laplacian eigenvectors.
- Source and target weighted degrees.

The combined `pair_features()` function returns these values in a `PairFeatures` dataclass. Current dense implementations are appropriate for toy graphs and small validation slices only. PrimeKG-scale analysis requires sparse approximations validated in `EXP-003`.

### Perturbation models and robust scoring

The implemented perturbation API includes:

- `edge_dropout()`, which randomly removes edges while optionally preserving protected relation types.
- `relation_dropout()`, which removes all edges from selected relation families.
- `weight_noise()`, which applies multiplicative log-normal edge-weight noise.
- `summarize_robustness()`, which evaluates a scorer over perturbed graph realizations and reports mean score, variance, and `mean_score - beta * variance`.
- `summarize_score_robustness()`, which performs analogous aggregation for tuple-edge experiments over multiple candidate pairs.

Planned perturbation extensions include neighborhood masking, degree-preserving rewiring, and biological-layer dropout.

### Splits and negative sampling

The planned benchmark will use fixed split artifacts for random edge, disease-held-out, drug-held-out, and disease-class-held-out settings. Random edge splits are treated as a sanity check only. Out-of-distribution claims require disease-held-out, drug-held-out, or disease-class-held-out evaluation with leakage checks. The current split bundle removes validation/test positive label edges from the graph used for scoring and emits leakage reports. Negative sampling will include degree-matched negatives and at least one alternate scheme because unlabeled drug-disease non-edges are not confirmed negatives.

### Models and evaluation

The planned model ladder includes degree and network proximity baselines (`EXP-030`), KG embedding baselines (`EXP-031`), one heterogeneous GNN baseline if feasible (`EXP-032`), spectral-only models (`EXP-050`), hybrid models combining baseline embeddings or scores with spectral features (`EXP-051`), and a disorder-aware reranker (`EXP-052`). Metrics include AUROC, AUPRC, Hits@K, Recall@K, MRR, expected calibration error, robust AUPRC, degradation slope under perturbation, and score-variance summaries.

### Figure generation

All figures should be regenerated from stored artifacts rather than ad hoc notebooks. Figure-specific inputs, expected panels, and owning experiments are listed in [figures_plan.md](/home/tibor/code_repos/drug_repurposing/paper/figures_plan.md).

## Limitations

No PrimeKG ingestion, benchmark split, baseline comparison, perturbation sweep, reranker, or case-study analysis has been run at the time of this skeleton. The implemented spectral routines are dense and only suitable for toy graphs unless replaced or complemented by sparse approximations. Current perturbations cover edge dropout, relation dropout, and weight noise, but not all planned disorder families. Negative sampling may strongly influence apparent performance because unlabeled drug-disease pairs are not validated negatives. Knowledge-graph edges may encode publication bias, database curation bias, and clinical indication leakage. External validation against DRKG or non-graph evidence is planned but optional and not yet available.

## Data Availability

The current repository contains toy graph definitions and code for small-scale experiments. Planned PrimeKG analyses will require raw PrimeKG source files, normalized entity/relation/edge tables, fixed split artifacts, prediction tables, perturbation score distributions, and figure-ready derived tables. Public release should include all non-restricted processed artifacts needed to reproduce figures, plus hashes or version identifiers for any raw data that cannot be redistributed.

## Code Availability

The analysis code is being developed in this repository under `src/drug_repurposing`. At the time of this skeleton, the available API covers toy heterogeneous graphs, sparse adjacency construction, normalized Laplacians, small-graph heat and resolvent features, low-frequency embeddings, edge/relation/weight perturbations, and robustness summaries. A release-ready manuscript will require scripted experiment entry points for all registered experiments, fixed configurations, environment metadata, and figure-generation scripts.

## Acknowledgements

TODO: Add funding, compute, dataset, and contributor acknowledgements when the project context is finalized.

## Author Contributions

TODO: Add author contribution taxonomy once authorship is known.

## Competing Interests

TODO: State competing interests or declare none.

## References

TODO: Add formal references for PrimeKG, DRKG, Therapeutics Data Commons, TxGNN, network medicine baselines, knowledge-graph embedding baselines, heterogeneous GNN baselines, graph heat kernels, resolvent methods, and perturbation robustness.
