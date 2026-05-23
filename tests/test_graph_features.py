import numpy as np

from drug_repurposing.graph_features import (
    build_weighted_adjacency,
    heat_kernel_pair_features,
    normalized_laplacian,
)


def toy_heterogeneous_edges():
    return [
        ("drug_a", "targets", "protein_x", 1.0),
        ("drug_b", "targets", "protein_y", 1.0),
        ("protein_x", "interacts", "protein_y", 0.5),
        ("protein_x", "participates_in", "pathway_m", 1.0),
        ("protein_y", "participates_in", "pathway_m", 1.0),
        ("pathway_m", "associated_with", "disease_alpha", 1.0),
        ("protein_y", "associated_with", "disease_beta", 0.75),
    ]


def toy_node_order():
    return [
        "drug_a",
        "drug_b",
        "protein_x",
        "protein_y",
        "pathway_m",
        "disease_alpha",
        "disease_beta",
    ]


def test_normalized_laplacian_is_symmetric_with_expected_diagonal():
    adjacency = build_weighted_adjacency(
        edges=toy_heterogeneous_edges(),
        node_order=toy_node_order(),
        relation_weights={
            "targets": 1.0,
            "interacts": 0.8,
            "participates_in": 0.6,
            "associated_with": 1.2,
        },
        undirected=True,
    )

    laplacian = normalized_laplacian(adjacency)

    dense_laplacian = laplacian.toarray()
    assert dense_laplacian.shape == (7, 7)
    assert np.allclose(dense_laplacian, dense_laplacian.T)
    assert np.allclose(np.diag(dense_laplacian), np.ones(7))

    eigenvalues = np.linalg.eigvalsh(dense_laplacian)
    assert eigenvalues.min() >= -1e-10
    assert eigenvalues.max() <= 2.0 + 1e-10


def test_heat_kernel_pair_features_have_stable_shape_and_range():
    adjacency = build_weighted_adjacency(
        edges=toy_heterogeneous_edges(),
        node_order=toy_node_order(),
        undirected=True,
    )
    laplacian = normalized_laplacian(adjacency)

    features = heat_kernel_pair_features(
        laplacian=laplacian,
        node_index={node: i for i, node in enumerate(toy_node_order())},
        pairs=[
            ("drug_a", "disease_alpha"),
            ("drug_b", "disease_beta"),
        ],
        taus=[0.1, 1.0, 3.0],
    )
    repeated_features = heat_kernel_pair_features(
        laplacian=laplacian,
        node_index={node: i for i, node in enumerate(toy_node_order())},
        pairs=[
            ("drug_a", "disease_alpha"),
            ("drug_b", "disease_beta"),
        ],
        taus=[0.1, 1.0, 3.0],
    )

    assert features.shape == (2, 3)
    assert np.isfinite(features).all()
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)
    assert np.allclose(features, repeated_features)

    close_scale, medium_scale, broad_scale = features[0]
    assert close_scale < medium_scale < broad_scale
