# PrimeKG Smoke Analysis

This directory is populated by:

```bash
.venv/bin/python scripts/run_primekg_smoke_analysis.py --nrows 50000
```

The smoke run is not the full benchmark. It checks that the public PrimeKG CSV
route can be read, converted into the internal graph representation, and
summarized with relation/type counts before running full-scale experiments.

Expected artifacts:

- `sample_edges.csv`: first 1,000 read edges for schema inspection.
- `positive_pairs.csv`: extracted drug-disease positive pairs from the read slice.
- `metrics.json`: row, node, edge, relation, type, and positive-pair counts.
