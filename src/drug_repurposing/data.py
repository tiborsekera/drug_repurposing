"""Data loading and split utilities for biomedical knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

import numpy as np
import pandas as pd

from drug_repurposing.graph import Edge, HeterogeneousGraph, Node

PRIMEKG_DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"


@dataclass(frozen=True)
class PairSplit:
    """Train/validation/test drug-disease pair split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_id: str


def download_primekg(destination: str | Path) -> Path:
    """Download the ready-to-use PrimeKG CSV from Harvard Dataverse."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(PRIMEKG_DATAVERSE_URL, destination)
    return destination


def read_primekg_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read PrimeKG while preserving mixed identifier columns as strings."""

    path_string = str(path)
    if path_string.startswith(("http://", "https://")):
        request = Request(path_string, headers={"User-Agent": "drug-repurposing-research/0.1"})
        with urlopen(request) as response:
            return pd.read_csv(response, low_memory=False, nrows=nrows, dtype=str)
    return pd.read_csv(path, low_memory=False, nrows=nrows, dtype=str)


def primekg_to_graph(frame: pd.DataFrame, directed: bool = False) -> HeterogeneousGraph:
    """Convert a PrimeKG-like edge table to `HeterogeneousGraph`.

    Expected columns follow PrimeKG naming: `x_id`, `x_type`, `y_id`, `y_type`,
    and either `display_relation` or `relation`.
    """

    required = {"x_id", "x_type", "y_id", "y_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing PrimeKG columns: {sorted(missing)}")

    relation_column = "display_relation" if "display_relation" in frame.columns else "relation"
    if relation_column not in frame.columns:
        raise ValueError("PrimeKG table must contain display_relation or relation")

    node_types: dict[str, str] = {}
    edges: list[Edge] = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        source = str(values["x_id"])
        target = str(values["y_id"])
        source_type = str(values["x_type"])
        target_type = str(values["y_type"])
        relation = str(values[relation_column])
        node_types.setdefault(source, source_type)
        node_types.setdefault(target, target_type)
        edges.append(Edge(source, target, relation, 1.0))

    nodes = [Node(node_id, node_type) for node_id, node_type in sorted(node_types.items())]
    return HeterogeneousGraph.from_records(nodes, edges, directed=directed)


def graph_to_pair_frame(
    graph: HeterogeneousGraph,
    positive_relations: set[str] | None = None,
    drug_type: str = "drug",
    disease_type: str = "disease",
) -> pd.DataFrame:
    """Extract labelled positive drug-disease pairs from graph edges."""

    positive_relations = positive_relations or {"indication", "off-label use"}
    node_types = graph.node_types
    rows = []
    for edge in graph.edges:
        source_type = node_types.get(edge.source)
        target_type = node_types.get(edge.target)
        if edge.relation not in positive_relations:
            continue
        if source_type == drug_type and target_type == disease_type:
            rows.append({"drug_id": edge.source, "disease_id": edge.target, "label": 1})
        elif source_type == disease_type and target_type == drug_type:
            rows.append({"drug_id": edge.target, "disease_id": edge.source, "label": 1})
    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)


def sample_negative_pairs(
    graph: HeterogeneousGraph,
    positive_pairs: pd.DataFrame,
    n_negatives: int,
    seed: int = 0,
    drug_type: str = "drug",
    disease_type: str = "disease",
) -> pd.DataFrame:
    """Sample unlabeled drug-disease non-edges as provisional negatives."""

    rng = np.random.default_rng(seed)
    node_types = graph.node_types
    drugs = np.array(sorted(node for node, typ in node_types.items() if typ == drug_type), dtype=object)
    diseases = np.array(sorted(node for node, typ in node_types.items() if typ == disease_type), dtype=object)
    positives = set(zip(positive_pairs["drug_id"], positive_pairs["disease_id"]))
    negatives: set[tuple[str, str]] = set()

    if len(drugs) == 0 or len(diseases) == 0:
        raise ValueError("Graph must contain drug and disease nodes")

    max_possible = len(drugs) * len(diseases) - len(positives)
    if n_negatives > max_possible:
        raise ValueError("Requested more negatives than available unlabeled pairs")

    while len(negatives) < n_negatives:
        drug = str(rng.choice(drugs))
        disease = str(rng.choice(diseases))
        pair = (drug, disease)
        if pair not in positives:
            negatives.add(pair)

    return pd.DataFrame(
        [{"drug_id": drug, "disease_id": disease, "label": 0} for drug, disease in sorted(negatives)]
    )


def random_pair_split(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    seed: int = 0,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> PairSplit:
    """Create a stratified random pair split."""

    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    parts = []
    for label, frame in [(1, positives), (0, negatives)]:
        shuffled = frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        n_total = len(shuffled)
        n_validation = int(round(n_total * validation_fraction))
        n_test = int(round(n_total * test_fraction))
        parts.append(
            (
                shuffled.iloc[: n_total - n_validation - n_test].assign(label=label),
                shuffled.iloc[n_total - n_validation - n_test : n_total - n_test].assign(label=label),
                shuffled.iloc[n_total - n_test :].assign(label=label),
            )
        )

    train = pd.concat([part[0] for part in parts], ignore_index=True)
    validation = pd.concat([part[1] for part in parts], ignore_index=True)
    test = pd.concat([part[2] for part in parts], ignore_index=True)
    return PairSplit(train=train, validation=validation, test=test, split_id=f"random_seed_{seed}")


def disease_held_out_split(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    held_out_diseases: set[str],
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> PairSplit:
    """Create a disease-held-out split with no held-out diseases in training."""

    all_pairs = pd.concat([positives.assign(label=1), negatives.assign(label=0)], ignore_index=True)
    test = all_pairs[all_pairs["disease_id"].isin(held_out_diseases)].reset_index(drop=True)
    remaining = all_pairs[~all_pairs["disease_id"].isin(held_out_diseases)].reset_index(drop=True)
    validation = remaining.sample(frac=validation_fraction, random_state=seed).reset_index(drop=True)
    train = remaining.drop(validation.index).reset_index(drop=True)
    return PairSplit(train=train, validation=validation, test=test, split_id="disease_held_out")
