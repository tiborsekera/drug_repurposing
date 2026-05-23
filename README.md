# Disorder-aware spectral graph learning for drug repurposing

This repository is the implementation workspace for the research direction in
`initial_brainstorm_results.md`.

The first milestone is an executable MVP:

1. represent a small heterogeneous biomedical knowledge graph,
2. build sparse graph operators,
3. compute spectral and transport-style drug-disease pair features,
4. perturb the graph with simple disorder models,
5. summarize prediction stability as a robustness signal.

Run the toy pipeline:

```bash
PYTHONPATH=src python3 -m drug_repurposing.experiments.toy_pipeline
```

Run tests:

```bash
python3 -m pytest
```
