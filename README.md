# Disorder-aware spectral graph learning for drug repurposing

This repository is the implementation workspace for the research direction in
`initial_brainstorm_results.md`.

The first milestone is an executable MVP:

1. represent a small heterogeneous biomedical knowledge graph,
2. build sparse graph operators,
3. compute spectral and transport-style drug-disease pair features,
4. perturb the graph with simple disorder models,
5. summarize prediction stability as a robustness signal.

The central research claim is not that spectral features alone improve drug
repurposing. The claim to test is whether disorder-response fingerprints expose
fragile high-confidence predictions after controlling for raw score and graph
degree.

Leakage control is mandatory: validation/test drug-disease label edges must be
removed from the graph used for feature computation. The `SplitGraphBundle` API
in `drug_repurposing.data` is the starting point for leakage-safe experiments.

Run the toy pipeline:

```bash
PYTHONPATH=src python3 -m drug_repurposing.experiments.toy_pipeline
```

Run tests:

```bash
python3 -m pytest
```
