# Experiment Registry

## Conventions

Experiment IDs use `EXP-###`. Each experiment must write a machine-readable config, immutable input references, prediction tables, aggregate metrics, and a short run note. All metrics should be reported per split, with random seeds recorded.

Common metrics:

- AUROC and AUPRC for ranking quality
- Hits@K, Recall@K, and MRR for candidate retrieval
- expected calibration error for confidence quality
- robust AUPRC and degradation slope under perturbation
- score variance, true-positive variance, and false-positive variance under disorder

Common prediction output schema:

`experiment_id`, `split_id`, `seed`, `drug_id`, `disease_id`, `label`, `score`, `rank`, `score_mean`, `score_var`, `features_uri`, `model_uri`.

## Registry

| ID | Stage | Purpose | Inputs | Outputs | Metrics |
| --- | --- | --- | --- | --- | --- |
| EXP-001 | S0 | Validate toy KG fixture and schema | Hand-built toy KG entities, relations, labels | Entity/relation tables, sparse adjacency, known positive pairs | Node/edge counts, connected components, deterministic checksum |
| EXP-002 | S1 | Compute exact spectral features on toy KG | EXP-001 graph, candidate drug-disease pairs | Pair feature table with eigen-distance, heat score, resolvent score, LDOS/IPR summaries | Feature non-null rate, runtime, rank of known positives |
| EXP-003 | S1 | Compare sparse approximations against exact toy features | EXP-001 graph, exact EXP-002 features | Approximate feature table and approximation error report | MAE/RMSE versus exact features, rank correlation, runtime |
| EXP-004 | S1 | Run MVP spectral-only link predictor on small KG slice | Small curated or sampled KG, fixed train/val/test pairs | Spectral-only model predictions and coefficients/importances | AUROC, AUPRC, Hits@K, MRR, calibration error |
| EXP-010 | S2 | Ingest and validate PrimeKG | Raw PrimeKG files, ingestion config | Normalized entity table, relation table, edge table, sparse matrices | Entity/relation counts, duplicate rate, isolated drug/disease count, component stats |
| EXP-011 | S2 | Build PrimeKG small development slice | EXP-010 graph, sampling config | Small KG slice with preserved drug-disease positives | Relation coverage, degree distribution comparison, retained positive count |
| EXP-020 | S3 | Create random edge split | EXP-010 or EXP-011 graph, negative sampling config | Train/val/test positives and negatives | Class balance, degree distribution, leakage checks |
| EXP-021 | S3 | Create disease-held-out split | EXP-010 graph, disease holdout list, negative sampling config | Zero-shot disease split artifacts | Held-out disease count, edge leakage checks, positive/negative counts |
| EXP-022 | S3 | Create drug-held-out split | EXP-010 graph, drug holdout list, negative sampling config | New-drug split artifacts | Held-out drug count, edge leakage checks, positive/negative counts |
| EXP-023 | S3 | Create disease-class-held-out split | EXP-010 graph, disease class annotations | Class-held-out split artifacts | Held-out class count, train/test class separation, positive/negative counts |
| EXP-030 | S4 | Run degree and network proximity baselines | Split artifacts EXP-020 to EXP-023, PrimeKG matrices | Baseline score tables | AUROC, AUPRC, Hits@K, MRR, degree-stratified AUPRC |
| EXP-031 | S4 | Run KG embedding baselines | Split artifacts, relation triples | TransE/RotatE/ComplEx predictions and embeddings | AUROC, AUPRC, Hits@K, MRR, calibration error |
| EXP-032 | S4 | Run heterogeneous GNN baseline | Split artifacts, typed graph tensors | R-GCN/HGT predictions and embeddings | AUROC, AUPRC, Hits@K, MRR, calibration error |
| EXP-040 | S5 | Edge-dropout robustness sweep | Trained baseline or spectral model, split artifacts, dropout strengths | Prediction distributions across perturbed graphs | Robust AUPRC, degradation slope, score variance |
| EXP-041 | S5 | Relation-dropout sensitivity | Trained model, relation-specific matrices, split artifacts | Per-relation ablation predictions | Delta AUPRC, delta Hits@K, relation importance, score variance |
| EXP-042 | S5 | Weight-noise robustness sweep | Weighted graph, trained model, noise strengths | Prediction distributions under edge-weight noise | Robust AUPRC, degradation slope, calibration error |
| EXP-043 | S5 | Drug/disease neighborhood masking | Trained model, split artifacts, masking strengths | Masked-neighborhood prediction distributions | Robust AUPRC, score variance, recall loss by degree bin |
| EXP-044 | S5 | Degree-preserving rewiring control | PrimeKG graph, rewiring config, trained model | Rewired-graph predictions and null distributions | Null AUPRC, degree-bias gap, empirical p-values |
| EXP-045 | S5 | Biological-layer dropout | Relation-to-layer map, trained model, split artifacts | Layer-ablation predictions | Delta AUPRC, delta Hits@K, layer sensitivity |
| EXP-050 | S6 | Train spectral-only full split models | Spectral features, EXP-020 to EXP-023 splits | Spectral model predictions and feature importances | AUROC, AUPRC, Hits@K, MRR, calibration error |
| EXP-051 | S6 | Add spectral features to embedding/GNN baseline | Baseline embeddings/scores, spectral feature tables | Hybrid model predictions | Metric delta versus baseline, calibration delta |
| EXP-052 | S6 | Train disorder-aware reranker | Baseline scores, spectral features, disorder mean/variance features | Reranked candidate lists and robust scores | Robust AUPRC, degradation slope, Hits@K, calibration error |
| EXP-053 | S6 | Optional disorder-regularized model | Baseline model, perturbation sampler, variance penalty config | Regularized model predictions | Robust AUPRC, score variance, clean-performance tradeoff |
| EXP-060 | S7 | Generate leaderboard tables | Outputs from EXP-030 to EXP-053 | Split-wise model comparison tables | Best and delta metrics, confidence intervals |
| EXP-061 | S7 | Generate robustness phase diagram | Outputs from EXP-040 to EXP-045 and EXP-052 | AUPRC/Hits@K versus disorder strength plots | Degradation slope and area under robustness curve |
| EXP-062 | S7 | Analyze stability of true versus false positives | Prediction distributions from perturbation experiments | Variance histograms, calibration bins, separation plots | TP/FP variance ratio, AUROC using stability alone |
| EXP-063 | S7 | Produce case studies | Top robust/spurious predictions, relation-channel sensitivity, stable paths | Case-study tables and path summaries | Rank, score mean/variance, relation sensitivity, path recurrence |

## Minimum Viable Experiment Sequence

Run this sequence before attempting the full paper pipeline:

1. `EXP-001` toy KG fixture.
2. `EXP-002` exact spectral features.
3. `EXP-004` toy/small KG spectral-only predictor.
4. `EXP-010` PrimeKG ingestion.
5. `EXP-021` disease-held-out split.
6. `EXP-030` network proximity baseline.
7. `EXP-040` edge-dropout robustness sweep.
8. `EXP-052` disorder-aware reranker.
9. `EXP-061` robustness phase diagram.

This is enough to test the core claim that perturbation stability adds information beyond a raw link-prediction score.

## Artifact Expectations

Each experiment should write:

- `config.yaml` with data versions, seeds, model settings, and perturbation settings
- `inputs.json` with hashes or paths for upstream artifacts
- `predictions.parquet` or equivalent prediction table
- `metrics.json` with aggregate and stratified metrics
- `run.md` with a short interpretation and known caveats

Stored artifacts should allow figures to be regenerated without retraining unless the experiment explicitly changes data, splits, or model parameters.

