# Literature Context: Disorder-Aware Spectral Graph Learning for Drug Repurposing

## Purpose

This note positions the project for a Nature-family paper attempt. It is based on `initial_brainstorm_results.md`, `docs/roadmap.md`, and `docs/experiment_registry.md`. No external verification was performed during this write-up. Any current facts, benchmark status, dataset sizes, citation metadata, or leaderboard claims should be verified externally by the parent before submission or public release.

## Working Position

The project should not be framed as "another GNN for drug repurposing." TxGNN already occupies the high-visibility position of a graph foundation model for zero-shot drug repurposing over a large medical knowledge graph, including disease generalization. PrimeKG and DRKG already provide biomedical graph infrastructure, while TDC provides standardized therapeutic ML tasks and evaluation tooling. KG embeddings and heterogeneous GNNs are mature enough that a simple performance-only link-prediction paper would be weak unless it demonstrates a large, reproducible, clinically meaningful advance.

The defensible gap is different:

> Biomedical knowledge-graph drug-repurposing models can produce confident rankings on incomplete, biased, and noisy graphs, but the field lacks a rigorous perturbation-stability framework for testing whether a prediction is structurally robust.

The contribution should therefore be a disorder-aware spectral and transport framework that quantifies when drug-disease predictions are stable or fragile under graph perturbation. The core scientific claim is about robustness, calibration, out-of-distribution behavior, and interpretability of graph-based predictions, not direct therapeutic discovery.

## Positioning Against Key Reference Points

### TxGNN

TxGNN is the main conceptual comparator because it explicitly targets zero-shot drug repurposing using a medical knowledge graph. The project should treat TxGNN as evidence that the field has moved beyond ordinary random-split link prediction and toward disease-held-out generalization.

The paper should position itself as complementary:

- TxGNN-style work asks whether a graph model can rank candidates for unseen or poorly served diseases.
- This project asks whether high-ranked candidates remain stable when the underlying biomedical graph is perturbed.
- A direct TxGNN reproduction would be valuable, but the first defensible version can use accessible KG embedding and GNN baselines if all models share fixed splits, perturbations, and metrics.
- The strongest claim would show that disorder features or a disorder-aware reranker improve calibration, robust AUPRC, false-positive separation, or degradation slope on disease-held-out and drug-held-out splits, even when raw AUROC/AUPRC gains are modest.

Avoid implying that the method supersedes TxGNN unless it is reproduced under matched data, splits, labels, negative sampling, and compute budget.

### PrimeKG

PrimeKG is the most natural primary graph because it was designed for precision-medicine-oriented biomedical knowledge integration across biological scales. In this project it should be treated as the main experimental substrate, not as a contribution by itself.

Required positioning:

- PrimeKG supplies the heterogeneous biomedical graph.
- The project contributes graph operators, spectral/transport descriptors, disorder perturbation regimes, and stability analyses on top of that graph.
- All PrimeKG ingestion statistics must be reported from the local pipeline: entity counts, relation counts, duplicate handling, connected components, drug/disease coverage, isolated nodes, and relation-family coverage.
- Any size or composition facts copied from the brainstorm or source pages must be rechecked externally before publication.

### DRKG

DRKG should be treated as an external replication graph if time and compute allow. It is useful because it differs from PrimeKG in schema, relation inventory, provenance, and graph density.

The right use is not to tune on DRKG and claim a new best model. The right use is to test whether the same qualitative stability findings survive a second biomedical KG:

- robust predictions degrade more slowly under perturbation;
- unstable high-score predictions are enriched for false positives or poorly calibrated bins;
- relation/layer dropout reveals dependence on narrow evidence channels;
- degree-preserving rewiring reduces or destroys the apparent signal if the model is mainly exploiting graph bias.

If DRKG is not included, the paper must state that external graph replication remains future work.

### Therapeutics Data Commons

TDC should be used for benchmark discipline where it fits the task. The value is standardization: reproducible splits, metrics, data processors, and task framing. If the project uses custom PrimeKG splits, TDC still matters as a reference for expected evaluation hygiene.

The project should not overclaim "TDC benchmark superiority" unless it runs official tasks exactly as specified. If it uses TDC only as tooling or inspiration, say that plainly.

### KG Embedding Baselines

The minimum baseline set should include TransE, RotatE, and ComplEx if feasible. These are not cutting-edge enough to be the only comparison, but they are essential controls because they expose whether the proposed features add value beyond standard relational embedding geometry.

The comparison should answer:

- Does the spectral/disorder feature set improve a KG embedding model under held-out disease or held-out drug splits?
- Does the improvement persist under degree-matched negatives and alternate negative sampling?
- Does perturbation variance add information beyond the raw embedding score?
- Are gains still visible after controlling for node degree, relation frequency, and training-edge proximity?

Report clean-performance metrics and robustness metrics separately. A model that wins only after perturbation but loses heavily on unperturbed ranking needs a narrower claim.

### Heterogeneous GNN Baselines

At least one relation-aware or heterogeneous GNN baseline should be included if compute permits, such as R-GCN or HGT. HAN may be relevant if the implementation uses metapath-style views, but it is less natural if the project keeps the full typed KG.

The GNN comparison should not be framed as "spectral methods versus GNNs." The stronger framing is "spectral/transport and disorder descriptors are complementary model diagnostics or reranking features for graph-learning systems."

Good model ladder:

1. degree and network-proximity controls;
2. KG embeddings: TransE, RotatE, ComplEx;
3. heterogeneous GNN baseline: R-GCN or HGT;
4. spectral-only pair scorer;
5. baseline plus spectral/transport features;
6. disorder-aware reranker using score mean and variance under perturbation;
7. optional disorder-regularized GNN.

## Methodological Gap

Many drug-repurposing KG studies optimize ranking quality on observed drug-disease links. The central weakness is that observed graph edges are neither complete nor uniformly reliable. Literature-derived KGs overrepresent well-studied diseases, highly connected proteins, popular drugs, and heavily curated pathways. Random edge splits can leak local neighborhoods and inflate performance. Negative samples are usually unlabeled non-edges rather than confirmed negatives.

This project addresses that gap by treating graph uncertainty as an experimental variable. It asks whether predictions are stable across plausible graph perturbations:

- edge dropout for missing evidence;
- relation dropout for dependence on one evidence source or relation family;
- edge-weight noise for confidence uncertainty;
- drug/disease neighborhood masking for sparse evidence regimes;
- biological-layer dropout for cross-scale dependency;
- degree-preserving rewiring as a degree-bias null model.

The spectral framing adds interpretable descriptors:

- low-frequency proximity in a normalized graph operator;
- heat-kernel diffusion between drug and disease nodes;
- resolvent or Green's-function response as a transport-like score;
- local spectral density and inverse participation summaries;
- perturbation variance and relation-channel sensitivity.

The physics analogy must stay methodological. The paper can discuss graph operators, spectra, diffusion, transport analogies, and disorder ensembles. It should not imply quantum biology or literal physical transport in biomedical systems.

## Evaluation Requirements

The evaluation must be designed to survive skeptical review:

- fixed train/validation/test artifacts with immutable IDs;
- random split reported only as a sanity check;
- disease-held-out split for zero-shot disease generalization;
- drug-held-out split for new-drug generalization;
- disease-class-held-out split if labels support it;
- degree-matched negatives plus at least one alternate negative sampling scheme;
- leakage checks showing that held-out drugs/diseases/classes are not indirectly used as positive labels during training;
- metrics including AUROC, AUPRC, Hits@K, Recall@K, MRR, expected calibration error, robust AUPRC, degradation slope, and score variance;
- confidence intervals or repeated-seed uncertainty;
- stratification by node degree, relation coverage, disease class, and evidence density.

The key figures should include:

- clean predictive performance across splits;
- AUPRC or Hits@K as a function of disorder strength;
- score variance for true positives versus false positives;
- calibration curves before and after disorder-aware reranking;
- relation/layer dropout sensitivity;
- case studies showing stable and unstable predictions with path or channel evidence.

## Claims to Make Only If Supported

These are plausible target claims:

- Disorder-aware spectral features improve robustness of biomedical KG drug-disease ranking under graph perturbation.
- Perturbation variance provides signal beyond the raw link-prediction score.
- Stability-aware reranking improves calibration or false-positive separation on held-out disease and/or held-out drug splits.
- Relation-channel and biological-layer dropout identify predictions that depend on narrow evidence sources.
- Spectral/transport descriptors provide interpretable diagnostics complementary to KG embeddings or heterogeneous GNNs.

These claims require quantitative support on fixed splits and should be stated with the exact metrics and datasets used.

## Claims to Avoid

Avoid these unless there is substantially stronger evidence:

- "We discovered new drugs" or "we identify clinically actionable therapies" without external temporal validation, expert review, clinical evidence, or experimental validation.
- "Nature-grade drug discovery" based only on retrospective KG link prediction.
- "Outperforms TxGNN" without a matched, reproduced TxGNN comparison.
- "Generalizes to all biomedical KGs" if only PrimeKG is used.
- "Robust to graph noise" if only random edge dropout is tested.
- "Biologically causal" based on graph paths or spectral features alone.
- "Unbiased" or "fair" unless study bias, curation bias, disease prevalence, and publication bias are explicitly modeled.
- "Negative predictions are true contraindications" when negatives are sampled non-edges.
- "Quantum transport" or "topological phase" language as if the KG were a physical quantum system.

## Compact Citation List

The parent should verify all metadata and current URLs externally before citation freeze.

- TxGNN / graph foundation model for clinician-centered drug repurposing: https://www.nature.com/articles/s41591-024-03233-x
- PrimeKG project page: https://zitniklab.hms.harvard.edu/projects/PrimeKG/
- DRKG repository: https://github.com/gnn4dr/DRKG
- Therapeutics Data Commons: https://tdcommons.ai/
- Baseline families to cite after external verification: TransE, RotatE, ComplEx, R-GCN, HGT, HAN, network medicine proximity methods, graph robustness/uncertainty methods, and spectral graph learning/heat-kernel methods.

