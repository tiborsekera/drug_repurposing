## Project thesis

**Working title:**
**Disorder-aware spectral graph learning for robust drug repurposing**

Core claim:

> Biomedical knowledge-graph drug-repurposing models are sensitive to graph incompleteness, noisy edges, and biased relation coverage. Spectral and transport-style graph features can quantify this sensitivity and improve robustness, calibration, and interpretability under out-of-distribution disease/drug splits.

This is the right framing because TxGNN already occupies the “graph foundation model for zero-shot drug repurposing” position, using a medical knowledge graph to rank candidate indications and contraindications across 17,080 diseases. A publishable contribution should therefore not be “another GNN for drug repurposing,” but a robustness/interpretablity layer that tests where such graph predictions are stable or fragile. ([Nature][1])

Your thesis gives a real methodological bridge: it used graph/lattice Hamiltonians, sparse operators, scattering/transmission ideas, and disorder-driven phase behavior; the topological Anderson section is especially relevant because it studied how disorder renormalizes a system and changes its observable phase. 

---

# 1. Research question

Frame the paper around three questions:

### Q1 — Predictive value

Do spectral/transport descriptors on biomedical knowledge graphs improve drug-disease link prediction, especially under **disease-held-out** and **drug-held-out** splits?

### Q2 — Robustness

Are correct drug-repurposing predictions more stable under graph perturbations than false positives?

### Q3 — Interpretability

Can disorder sensitivity, spectral localization, and transport paths explain why a drug-disease prediction is robust or spurious?

The non-obvious angle: the paper does not need to beat TxGNN globally to be publishable. It needs to show that **prediction confidence without perturbation stability is incomplete**.

---

# 2. Dataset strategy

Use **PrimeKG** as the primary graph. It is suitable because it integrates 20 biomedical resources and contains 17,080 diseases, 7,957 drugs, and roughly 4 million relationships across biological scales. ([Zitnik Lab][2])

Use **DRKG** as an external replication graph if time permits. DRKG relates genes, compounds, diseases, biological processes, side effects, and symptoms, with 97,238 entities and 5,874,261 triplets across 107 edge types. ([GitHub][3])

Use **Therapeutics Data Commons** for standardized task framing, splits, metrics, and possible benchmark integration. TDC provides AI-ready datasets, curated benchmarks, data processors, meaningful data splits, and a Python library. ([TDC][4])

Priority order:

| Priority | Dataset            | Purpose                                |
| -------: | ------------------ | -------------------------------------- |
|        1 | PrimeKG            | Main biomedical KG                     |
|        2 | TxGNN-style splits | Zero-shot / disease-held-out benchmark |
|        3 | TDC task wrapper   | Standardized evaluation                |
|        4 | DRKG               | External robustness replication        |

---

# 3. Method: define a biomedical “Hamiltonian”

Represent the biomedical KG as one or more sparse operators.

### Basic operators

For a heterogeneous KG with nodes (V), relations (R), and weighted adjacency matrices (A_r):

[
H = \sum_{r \in R} \alpha_r A_r
]

or use a normalized Laplacian:

[
L = I - D^{-1/2} A D^{-1/2}
]

For relation-aware modeling:

[
H_{\text{rel}} = \sum_r \alpha_r A_r
]

where (\alpha_r) can be learned, fixed, or estimated by validation.

This is the direct translation of your thesis-style thinking: construct an operator, perturb it, study spectra and transport observables.

---

# 4. Core features to compute

For each candidate drug-disease pair ((d, z)), compute features from graph operators.

| Feature family               | Concrete feature                      | Interpretation                                      |
| ---------------------------- | ------------------------------------- | --------------------------------------------------- |
| Spectral proximity           | Low-frequency embedding distance      | Shared global biological neighborhood               |
| Heat kernel                  | ((e^{-\tau L})_{dz})                  | Diffusive reachability at scale (\tau)              |
| Resolvent / Green’s function | (((\lambda I - H)^{-1})_{dz})         | Transport-like response between drug and disease    |
| Local density of states      | Spectral mass near drug/disease nodes | Whether nodes sit in dense functional modules       |
| Localization                 | Inverse participation ratio           | Whether prediction depends on localized graph modes |
| Disorder sensitivity         | Variance of score under perturbations | Robustness of prediction                            |
| Relation-channel sensitivity | Drop one relation type and rescore    | Which biological evidence class matters             |
| Path stability               | Stable high-contribution paths        | Explanation consistency                             |

The resolvent/Green’s-function feature is the most thesis-specific. It is close in spirit to scattering/transport: perturbation at one region, response at another.

---

# 5. Disorder model

This is the paper’s distinctive contribution.

Apply perturbations to the KG and evaluate whether predictions survive.

### Disorder types

| Disorder type              | Mechanism                                                 | Biomedical analogue                      |
| -------------------------- | --------------------------------------------------------- | ---------------------------------------- |
| Edge dropout               | Randomly remove edges                                     | Missing literature/database evidence     |
| Relation dropout           | Remove one relation family                                | Dependency on one evidence source        |
| Weight noise               | Perturb edge weights                                      | Confidence uncertainty                   |
| Node neighborhood masking  | Hide part of drug/disease neighborhood                    | Sparse evidence disease/drug             |
| Degree-preserving rewiring | Randomize graph while preserving degree                   | Tests whether model exploits degree bias |
| Biological-scale dropout   | Remove protein, phenotype, pathway, or side-effect layers | Tests cross-scale dependency             |

Define robustness score:

[
R(d,z) = \mathbb{E}*{\xi}[s*\xi(d,z)] - \beta \operatorname{Var}*{\xi}[s*\xi(d,z)]
]

where (s_\xi(d,z)) is the score under disorder realization (\xi).

This gives you a direct “disorder-aware” ranking: not only high predicted score, but high score with low perturbation variance.

---

# 6. Model variants

Build the paper around a clear ablation ladder.

| Model                      | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| Network proximity baseline | Classical network medicine baseline        |
| TransE / RotatE / ComplEx  | KG embedding baselines                     |
| R-GCN / HGT / HAN          | Heterogeneous GNN baselines                |
| Spectral-only model        | Tests your physics-inspired features alone |
| GNN + spectral features    | Tests feature complementarity              |
| Disorder-regularized GNN   | Main proposed model                        |
| Disorder-aware reranker    | Lightweight publishable version            |

The most feasible main model:

[
\text{score}(d,z) =
f_\theta[
h_d, h_z,
\phi_{\text{spectral}}(d,z),
\phi_{\text{transport}}(d,z),
\phi_{\text{disorder}}(d,z)
]
]

where (h_d, h_z) are GNN/KG embeddings and (\phi) are your spectral-disorder features.

A stronger but harder version adds a regularization term:

[
\mathcal{L}
===========

\mathcal{L}*{\text{link}}
+
\lambda \operatorname{Var}*{\xi}(s_\xi(d,z))
]

This penalizes predictions that are unstable under graph perturbation.

---

# 7. Evaluation design

The evaluation must avoid the classic weak drug-repurposing trap: random splits that leak biological neighborhoods.

Use four splits:

| Split                        | Why it matters                         |
| ---------------------------- | -------------------------------------- |
| Random edge split            | Sanity check only                      |
| Disease-held-out split       | Tests zero-shot disease generalization |
| Drug-held-out split          | Tests new-drug generalization          |
| Disease-class-held-out split | Hardest and most publishable           |

Metrics:

| Metric                                     | Use                                 |
| ------------------------------------------ | ----------------------------------- |
| AUROC                                      | General ranking quality             |
| AUPRC                                      | More relevant under class imbalance |
| Hits@K / Recall@K                          | Practical top-candidate ranking     |
| MRR                                        | Link-prediction quality             |
| Calibration error                          | Whether confidence is meaningful    |
| Robust AUPRC under disorder                | Your key metric                     |
| Score variance for true vs false positives | Your diagnostic contribution        |

Key plot:

> AUPRC versus disorder strength.

If your model degrades more slowly than baselines, the paper has a clear story.

---

# 8. Main figures for the paper

Aim for 6 core figures.

### Figure 1 — Concept

Biomedical KG → graph operator → spectral/transport features → disorder ensemble → robust drug ranking.

### Figure 2 — Operator construction

Show relation-specific adjacency matrices combined into a biomedical Hamiltonian/Laplacian.

### Figure 3 — Predictive performance

Compare baselines across random, disease-held-out, and drug-held-out splits.

### Figure 4 — Robustness phase diagram

Performance as function of disorder strength and disorder type.

This is where your thesis identity shows up: a phase-diagram-style result, not just a leaderboard table.

### Figure 5 — Stability separates true from false positives

Show that true positives have lower score variance under perturbation.

### Figure 6 — Case studies

Pick 3–5 drug-disease predictions and show stable paths, relation-channel dependence, and disorder sensitivity.

---

# 9. Minimum publishable claim

The safest paper claim:

> Disorder-aware spectral features improve robustness and calibration of biomedical KG drug-repurposing predictions under graph perturbation and out-of-distribution splits.

Do **not** claim:

> We discovered new drugs.

Unless you validate against later clinical evidence, external databases, or expert review.

A stronger but still defensible claim:

> Perturbation stability is an independent signal for drug-disease link reliability.

That is more publishable than a marginal performance bump.

---

# 10. Paper structure

## Abstract

Problem: KG drug repurposing is promising but fragile under incomplete/noisy biomedical graphs.
Method: disorder-aware spectral graph learning.
Result: improved robustness/calibration/OOD performance.
Contribution: physics-inspired perturbation framework.

## Introduction

Position against TxGNN and biomedical KG models. Explain that existing models emphasize predictive ranking, while graph uncertainty and perturbation stability remain underdeveloped. TxGNN is the key comparison because it explicitly targets zero-shot drug repurposing on medical KGs. ([Nature][1])

## Related work

Cover:

* biomedical KGs: PrimeKG, DRKG,
* KG embedding and heterogeneous GNNs,
* drug repurposing ML,
* spectral graph methods,
* graph robustness and uncertainty,
* network medicine.

## Method

Define graph operators, spectral features, transport/resolvent features, disorder ensemble, and robust scoring.

## Experiments

Datasets, splits, baselines, metrics, hyperparameters.

## Results

Performance, robustness, ablations, calibration, case studies.

## Discussion

Interpret why disorder sensitivity matters. Explicitly state limitations: KG incompleteness, no wet-lab validation, possible database bias, negative sampling ambiguity.

---

# 11. Execution roadmap

## Phase 1 — Reproducible baseline, 2–3 weeks

Goal: load PrimeKG, define drug-disease prediction task, implement clean splits.

Deliverables:

* graph preprocessing pipeline,
* train/val/test splits,
* negative sampling strategy,
* baseline models,
* first AUROC/AUPRC/Hits@K table.

Hard requirement: no random-split-only evaluation.

## Phase 2 — Spectral/transport features, 3–4 weeks

Implement:

* Laplacian eigenfeatures,
* heat-kernel approximation,
* resolvent approximation,
* local density of states,
* inverse participation ratio,
* drug-disease transport scores.

Use sparse methods. Do not compute full eigendecompositions on the full KG unless reduced/subgraph approximations are necessary.

## Phase 3 — Disorder ensemble, 3–4 weeks

Implement perturbation regimes:

* edge dropout,
* relation dropout,
* weight noise,
* degree-preserving rewiring,
* biological-scale dropout.

Output:

* prediction mean,
* prediction variance,
* robust score,
* degradation curves.

## Phase 4 — Model integration, 2–3 weeks

Train:

* spectral-only model,
* GNN-only model,
* GNN + spectral features,
* GNN + disorder regularization,
* disorder-aware reranker.

The reranker is likely the best first paper path because it can wrap existing models without needing to beat foundation models end-to-end.

## Phase 5 — Paper, 3–4 weeks

Write while experiments are running.

Deliverables:

* arXiv-style paper,
* reproducible GitHub repo,
* environment file,
* data-preprocessing scripts,
* fixed splits,
* model cards for each baseline,
* all figures scripted.

Total realistic timeline: **13–18 weeks** for a serious preprint.

---

# 12. Critical risks

### Risk 1 — Spectral features do not improve headline metrics

This is plausible. Mitigation: make robustness/calibration the primary contribution, not raw AUPRC.

### Risk 2 — Negative sampling dominates results

Drug-disease non-edges are not true negatives. Use multiple negative sampling schemes and report sensitivity.

### Risk 3 — Degree bias masquerades as biology

High-degree drugs and diseases may dominate. Include degree-preserving rewiring and degree-matched negative samples.

### Risk 4 — Too much physics analogy, not enough biomedical validity

Avoid claiming “quantum transport in biology.” This is graph-operator methodology, not quantum biology.

### Risk 5 — TxGNN comparison may be hard

You can compare conceptually and against accessible baselines first. A direct TxGNN reproduction is valuable but not mandatory for the first version if your evaluation focuses on perturbation robustness.

---

# 13. Best first paper formulation

Most publishable version:

> **Disorder-aware spectral reranking improves robustness of biomedical knowledge-graph drug repurposing under graph perturbation and zero-shot disease splits.**

This is narrow enough to execute and broad enough to matter.

Core contribution list:

1. A disorder model for biomedical knowledge graphs.
2. Spectral/transport features for drug-disease candidate scoring.
3. A robust reranking method applicable to existing KG/GNN models.
4. OOD evaluation under disease-held-out and drug-held-out splits.
5. Stability-based interpretability for candidate predictions.

That is a legitimate paper if the experiments are rigorous.

[1]: https://www.nature.com/articles/s41591-024-03233-x?utm_source=chatgpt.com "A foundation model for clinician-centered drug repurposing"
[2]: https://zitniklab.hms.harvard.edu/projects/PrimeKG/?utm_source=chatgpt.com "Precision Medicine Oriented Knowledge Graph"
[3]: https://github.com/gnn4dr/DRKG?utm_source=chatgpt.com "Drug Repurposing Knowledge Graph (DRKG)"
[4]: https://tdcommons.ai/?utm_source=chatgpt.com "Therapeutics Data Commons - TDC"
