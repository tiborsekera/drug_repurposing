"""Build figure-ready toy data for Figure 1/2 placeholders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    figure_dir = REPO_ROOT / "figures" / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)

    adjacency = sparse.load_npz(REPO_ROOT / "artifacts" / "EXP-001" / "adjacency.npz")
    laplacian = sparse.identity(adjacency.shape[0], format="csr")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees, dtype=float)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    scale = sparse.diags(inv_sqrt)
    laplacian = laplacian - scale @ adjacency @ scale
    eigenvalues = np.linalg.eigvalsh(laplacian.toarray())

    pd.DataFrame({"eigenvalue": eigenvalues}).to_csv(figure_dir / "toy_laplacian_spectrum.csv", index=False)
    metrics = json.loads((REPO_ROOT / "artifacts" / "EXP-002" / "metrics.json").read_text())
    (figure_dir / "toy_feature_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Wrote figure data to {figure_dir}")


if __name__ == "__main__":
    main()
