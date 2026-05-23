"""Run a small PrimeKG ingestion smoke analysis.

This script reads the public PrimeKG CSV from Harvard Dataverse unless a local
CSV path is supplied. It intentionally supports `--nrows` so the ingestion path
can be checked quickly before committing to full-graph experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_repurposing.data import PRIMEKG_DATAVERSE_URL, graph_to_pair_frame, primekg_to_graph, read_primekg_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PRIMEKG_DATAVERSE_URL, help="PrimeKG CSV path or URL")
    parser.add_argument("--nrows", type=int, default=50000, help="Rows to read for smoke analysis")
    parser.add_argument("--outdir", default="outputs/primekg_smoke", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = read_primekg_csv(args.input, nrows=args.nrows)
    graph = primekg_to_graph(frame)
    positive_pairs = graph_to_pair_frame(graph)

    relation_column = "display_relation" if "display_relation" in frame.columns else "relation"
    metrics = {
        "source": args.input,
        "nrows_requested": args.nrows,
        "rows_read": int(len(frame)),
        "node_count": int(len(graph.nodes)),
        "edge_count": int(len(graph.edges)),
        "relation_count": int(frame[relation_column].nunique()),
        "node_type_count": int(len(set(frame["x_type"]) | set(frame["y_type"]))),
        "positive_drug_disease_pairs": int(len(positive_pairs)),
        "top_relations": frame[relation_column].value_counts().head(20).to_dict(),
        "top_x_types": frame["x_type"].value_counts().head(20).to_dict(),
        "top_y_types": frame["y_type"].value_counts().head(20).to_dict(),
    }

    frame.head(1000).to_csv(outdir / "sample_edges.csv", index=False)
    positive_pairs.to_csv(outdir / "positive_pairs.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
