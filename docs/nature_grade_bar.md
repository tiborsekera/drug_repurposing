# Nature-Grade Evidence Bar

## Scope

This note defines what evidence would be needed for a Nature-family submission attempt on disorder-aware spectral graph learning for drug repurposing. It is a bar-setting document, not a claim that the current project has met the bar. No browsing or external verification was performed for this note.

## Minimum Defensible Paper Claim

The safest serious claim is:

> Disorder-aware spectral and transport descriptors improve the robustness, calibration, or interpretability of biomedical knowledge-graph drug-repurposing predictions under out-of-distribution splits and graph perturbations.

This claim is narrower than drug discovery and broader than a leaderboard result. It is defensible if the method consistently adds value over standard baselines under disease-held-out and drug-held-out evaluation.

## Nature-Family Claim Bar

A Nature-family paper attempt would need evidence in at least five categories.

### 1. Methodological Novelty

The method must be more than applying a GNN to PrimeKG. It needs a crisp technical contribution:

- a formal disorder model for biomedical KGs;
- graph-operator construction for heterogeneous biomedical relations;
- spectral, heat-kernel, resolvent, local-density, localization, and perturbation-stability features;
- a disorder-aware score or reranker that can wrap existing models;
- interpretable relation-channel or biological-layer sensitivity analyses.

The novelty should be stated as "perturbation stability as a reliability signal for KG drug repurposing," not as a new biomedical KG or a new generic GNN.

### 2. Rigorous Benchmarking

The evidence must include:

- PrimeKG ingestion with reproducible entity/relation statistics;
- fixed split artifacts for random, disease-held-out, drug-held-out, and preferably disease-class-held-out settings;
- degree-matched negatives and at least one alternate negative sampling scheme;
- leakage audits for held-out diseases, drugs, classes, and labels;
- baselines including degree/network proximity, TransE, RotatE, ComplEx, and at least one heterogeneous GNN if feasible;
- explicit positioning against TxGNN, with direct reproduction only if the same data, splits, labels, and compute assumptions can be matched;
- TDC-compatible task framing or a clear explanation when custom splits are used.

Random edge splits alone are not enough. A strong random-split table can be included as a sanity check, but it cannot carry the paper.

### 3. Robustness as the Main Result

The key result must be a stability result, not just a clean AUPRC delta. Required analyses:

- degradation curves across edge dropout, relation dropout, neighborhood masking, weight noise, biological-layer dropout, and degree-preserving rewiring;
- robust AUPRC or robust Hits@K across perturbation strength;
- degradation slope or area under the robustness curve;
- score-variance comparison for true positives, false positives, and high-confidence false positives;
- calibration curves before and after disorder-aware reranking;
- ablation showing which spectral/transport/disorder features matter.

A Nature-grade story would show that perturbation stability separates reliable from fragile predictions in a way that raw model confidence does not.

### 4. External Validity

The strongest version needs at least one of:

- replication on DRKG or another independent biomedical KG;
- temporal validation against drug-disease evidence added after the graph snapshot;
- evaluation on an official or clearly documented TDC task;
- expert review of top robust and top unstable predictions;
- literature validation with pre-registered search rules;
- wet-lab or clinical collaboration, if the claim moves toward therapeutic discovery.

Without external validation, the paper should remain a methods paper about robustness and reliability of KG predictions.

### 5. Biological Interpretability and Case Studies

Case studies should be used carefully. They should illustrate model behavior, not serve as proof of treatment efficacy.

Each case study should report:

- baseline score and disorder-aware score;
- perturbation mean and variance;
- rank change after reranking;
- stable relation channels or biological layers;
- sensitivity to relation dropout;
- whether the prediction is driven by high-degree hubs;
- known evidence status and whether that evidence was present in the training graph.

Include negative or cautionary cases: high raw score but unstable under perturbation. These are essential for the paper's reliability argument.

## Evidence That Would Support Stronger Claims

The following evidence would justify stronger wording:

- temporal holdout where predictions made from an older graph are enriched for drug-disease links later supported by independent evidence;
- expert blinded review showing robust predictions are judged more plausible than unstable predictions at matched raw score;
- successful replication of stability effects on DRKG and PrimeKG;
- robust ranking improvements over a reproduced TxGNN-style baseline under matched zero-shot splits;
- prospective experimental or clinical validation for a small number of predictions.

Without these, avoid discovery language.

## Red-Line Claims

Do not make these claims from retrospective KG experiments alone:

- the method discovered new therapies;
- top-ranked candidates are clinically actionable;
- the model proves mechanism of action;
- graph paths establish causality;
- sampled non-edges are true negatives;
- the method is unbiased;
- the method beats TxGNN or TDC state of the art unless reproduced directly;
- the disorder model captures all real biomedical uncertainty;
- spectral transport features imply physical transport in disease biology.

## Minimum Submission Package

Before submission, the project should have:

- immutable data and split manifests;
- scripted ingestion checks;
- baseline prediction files in a common schema;
- perturbation score distributions for each candidate, not only aggregate metrics;
- reproducible figure scripts for the six main figure groups;
- run notes documenting failed or negative results;
- sensitivity analyses for negative sampling and degree bias;
- a limitations section that explicitly covers KG incompleteness, publication bias, curation bias, label noise, and absence of wet-lab validation.

## Go / No-Go Criteria

Go if:

- disease-held-out or drug-held-out robustness improves over strong baselines;
- stability adds signal beyond raw score and degree;
- calibration improves or high-confidence false positives are reduced;
- results are stable across seeds and negative samplers;
- at least one external validity analysis is completed or the claim is clearly limited to methods.

No-go for a Nature-family attempt if:

- gains appear only on random splits;
- perturbation stability is mostly a proxy for degree;
- results disappear under degree-matched negatives;
- only one weak baseline is used;
- the paper depends on drug-discovery claims without validation;
- case studies are cherry-picked and unsupported by systematic analysis.

## Bottom Line

The strongest version of the paper is not "we found drugs." It is:

> Graph-based drug-repurposing confidence is incomplete without perturbation stability, and disorder-aware spectral analysis provides a reproducible way to measure that stability.

That is the claim to defend experimentally.

