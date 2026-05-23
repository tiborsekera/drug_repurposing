# Research Implementation Roadmap

## Scope

Build a reproducible research pipeline for disorder-aware spectral graph learning on biomedical knowledge graphs. The minimum publishable claim is that perturbation stability and spectral/transport descriptors improve robustness, calibration, or interpretability of drug-disease link prediction, especially under held-out disease and held-out drug settings.

Primary graph: PrimeKG. Optional replication graph: DRKG. Early development should use toy and small extracted KGs before full PrimeKG ingestion.

## Milestones

| Stage | Milestone | Main Outputs | Exit Criteria |
| --- | --- | --- | --- |
| S0 | Project scaffolding and toy KG | Minimal schema, synthetic/toy KG fixture, experiment config convention | A toy graph can produce deterministic drug-disease candidates and labels |
| S1 | MVP spectral features on toy/small KG | Laplacian features, heat-kernel scores, approximate resolvent scores, local spectral summaries | Features compute in seconds on a toy graph and minutes on a small KG slice |
| S2 | PrimeKG ingestion | Raw-to-normalized entity/relation tables, sparse adjacency builders, drug/disease node maps | PrimeKG can be loaded reproducibly with relation statistics and sanity checks |
| S3 | Link prediction splits | Random edge, disease-held-out, drug-held-out, disease-class-held-out splits | Split files contain train/val/test positives, negatives, and leakage checks |
| S4 | Baseline comparisons | Network proximity, degree baseline, TransE/RotatE/ComplEx, R-GCN or HGT if feasible | Baselines produce AUROC, AUPRC, Hits@K, MRR, and calibration metrics on fixed splits |
| S5 | Perturbation and disorder experiments | Edge dropout, relation dropout, weight noise, neighborhood masking, degree-preserving rewiring, biological-layer dropout | Each model has score mean/variance and degradation curves across disorder strengths |
| S6 | Model integration | Spectral-only scorer, baseline plus spectral features, disorder-aware reranker, optional disorder-regularized GNN | Robust reranker improves degradation, calibration, or false-positive separation on OOD splits |
| S7 | Paper figures and release package | Scripted tables/figures, case studies, environment notes, fixed splits | Six core figures can be regenerated from stored experiment outputs |

## Staged Plan

### Stage S0: Toy KG and Reproducibility

Define a small heterogeneous KG with drugs, diseases, genes, pathways, side effects, and treatments. Keep the fixture small enough for exact eigendecomposition so approximations can be validated. Establish stable experiment IDs, config files, random seeds, and output directories before full-scale work.

### Stage S1: MVP Spectral Features

Implement graph operators for a homogeneous collapsed graph and relation-weighted graph:

- normalized Laplacian `L = I - D^-1/2 A D^-1/2`
- relation-weighted operator `H = sum_r alpha_r A_r`
- low-frequency drug/disease embedding distances
- heat-kernel score `(exp(-tau L))_drug,disease`
- approximate resolvent or Green's-function score
- local density and inverse participation summaries

Use exact methods on toy graphs and sparse approximations on small KG slices. The output should be pair-level feature tables that can feed a simple classifier.

### Stage S2: PrimeKG Ingestion

Normalize PrimeKG into typed entity and relation tables with stable integer IDs. Build sparse relation matrices and a collapsed adjacency. Track relation counts, connected components, degree distributions, duplicate edges, and isolated drug/disease nodes. Do not start model claims until ingestion checks are scripted.

### Stage S3: Link Prediction Splits

Create fixed split artifacts for:

- random edge split as a sanity check only
- disease-held-out split for zero-shot disease generalization
- drug-held-out split for new-drug generalization
- disease-class-held-out split for the hardest OOD setting

Use degree-matched negatives and at least one alternate negative sampling scheme. Report sensitivity to negative sampling because unlabeled drug-disease non-edges are not confirmed negatives.

### Stage S4: Baseline Comparisons

Start with fast baselines that expose graph bias:

- degree and common-neighbor style scores
- network proximity
- TransE, RotatE, and ComplEx
- one heterogeneous GNN baseline if compute permits

Every baseline should run on the same split files and emit the same prediction schema. Prioritize clean comparability over model variety.

### Stage S5: Perturbation and Disorder Experiments

Evaluate prediction stability under graph perturbations:

- edge dropout for missing evidence
- relation dropout for evidence-source dependence
- weight noise for confidence uncertainty
- drug/disease neighborhood masking for sparse evidence cases
- degree-preserving rewiring for degree-bias controls
- biological-layer dropout for protein, phenotype, pathway, or side-effect layer dependence

Store per-candidate score distributions, not just aggregate metrics. The central analysis is whether true positives degrade more slowly and show lower variance than false positives.

### Stage S6: Model Integration

Build the model ladder:

- spectral-only model over pair features
- baseline embedding model
- baseline plus spectral/transport features
- disorder-aware reranker using mean and variance under perturbation
- optional disorder-regularized model with a variance penalty

The likely first-paper path is the reranker because it can wrap existing scores and show that confidence without perturbation stability is incomplete.

### Stage S7: Paper Figures

Generate six scripted figure groups:

| Figure | Content | Required Data |
| --- | --- | --- |
| F1 | Concept pipeline: KG to operator to disorder ensemble to robust ranking | Method schematic inputs |
| F2 | Operator construction and relation channels | Relation matrices and graph statistics |
| F3 | Predictive performance across splits | Baseline and proposed model metrics |
| F4 | Robustness phase diagram | AUPRC/Hits@K versus disorder strength/type |
| F5 | Stability separates true and false positives | Score variance distributions and calibration bins |
| F6 | Case studies | Top predictions, stable paths, relation-channel sensitivity |

## Decision Gates

| Gate | Question | Continue If |
| --- | --- | --- |
| G1 | Do spectral features work on toy/small KGs? | Features are deterministic, interpretable, and non-degenerate |
| G2 | Is PrimeKG ingestion reliable? | Entity/relation counts and split leakage checks pass |
| G3 | Do splits expose OOD difficulty? | Held-out disease/drug scores are meaningfully harder than random splits |
| G4 | Does disorder stability add signal? | Stability metrics improve calibration, robust AUPRC, or false-positive separation |
| G5 | Is the paper claim defensible? | Results support robustness/interpretablity claims without relying on drug-discovery novelty |

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Spectral features do not improve headline AUPRC | Center the claim on robustness, calibration, and interpretation |
| Negative sampling dominates results | Use degree-matched and alternate negatives; report sensitivity |
| Degree bias masquerades as biology | Include degree baselines and degree-preserving rewiring |
| Full KG spectral methods are too expensive | Use sparse approximations, relation subsets, and neighborhood subgraphs |
| TxGNN comparison is hard to reproduce | Compare against accessible baselines first; position TxGNN as conceptual context unless reproduction is available |

