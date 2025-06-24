import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer, RobustScaler


def normalize_numerical_features(dataset, scaler=None):
    """
    Normalizes the numerical features in the dataset using Min-Max scaling.

    Parameters:
    - filtered_dataset: The dataset whose numerical features will be normalized.

    Returns:
    - filtered_dataset: The dataset with normalized numerical features.
    - scaler: The fitted MinMaxScaler object.
    """

    numerical_features = [data.features_numerical.numpy() for data in dataset]
    all_numerical_features = np.vstack(numerical_features)

    if scaler is None:
        scaler = QuantileTransformer()
        scaler.fit(all_numerical_features)
    else:
        scalar = scaler

    for data in dataset:
        data.features_numerical = torch.tensor(
            scaler.transform(data.features_numerical.numpy()), dtype=torch.float
        )

    return dataset, scaler

# Define the indices of the one-hot encoded materials to filter out
# material_group_mapping = {
#     0: "Unknown (0)",
#     1: "Steel (1)",
#     2: "Stainless Steel (2)",
#     3: "Aluminium (3)",
#     4: "Plastic (4)",
#     5: "Material Group 5",
#     6: "Material Group 6",
#     7: "Material Group 7",
#     8: "Material Group 8",
#     9: "Material Group 9",
#     10: "Material Group 10",
#     11: "Material Group 11",
#     12: "Insert (12)",
#     13: "procured part (13)",
#     14: "Articles (14)",
#     15: "Titanium (15)",
#     16: "Copper (16)",
#     17: "Brass (17)",
# }


def filter_categorical_features(dataset, filter_indices):
    """
    Filters out data points from the dataset that contain specified categorical feature indices.

    Parameters:
    - dataset: The dataset to filter.
    - filter_indices: List or array of indices of the categorical features to filter out.

    Returns:
    - filtered_dataset: The dataset after filtering out specified categorical features.
    """

    # Create a mask for the materials to filter out
    data_removed = np.zeros(len(dataset), dtype=bool)
    for i, data in enumerate(dataset):
        # Check if any of the specified material indices are present in the one-hot encoded features
        data_removed[i] = np.any(
            data.features_categorical.numpy()[:, filter_indices], axis=1
        ).any()

    filtered_dataset = [dataset[i] for i in range(len(dataset)) if not data_removed[i]]

    stats = {
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "num_removed": len(dataset) - len(filtered_dataset),
    }

    return filtered_dataset, stats



def filter_dataset_by_num_nodes(dataset, min_num_nodes=1, max_num_nodes=None):
    """
    Filters a dataset to remove graphs with number of nodes outside a given range.

    Parameters:
    - dataset: The dataset to filter.
    - min_num_nodes: Minimum allowed number of nodes (inclusive, default: 1).
    - max_num_nodes: Maximum allowed number of nodes (inclusive, default: None, no upper limit).

    Returns:
    - filtered_dataset: The dataset after filtering.
    - stats: A dictionary containing statistics about the filtering.
    """
    num_nodes_list = [data.num_nodes for data in dataset]
    if max_num_nodes is not None:
        mask = [(n >= min_num_nodes) and (n <= max_num_nodes) for n in num_nodes_list]
    else:
        mask = [n >= min_num_nodes for n in num_nodes_list]
    filtered_dataset = [d for d, keep in zip(dataset, mask) if keep]

    stats = {
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "num_removed": len(dataset) - len(filtered_dataset),
        "min_num_nodes": min_num_nodes,
        "max_num_nodes": max_num_nodes,
    }
    return filtered_dataset, stats

def filter_dataset_by_labels(dataset, min_label=0.01, max_label=None):
    """
    Filters a dataset to remove graphs with number of nodes outside a given range.

    Parameters:
    - dataset: The dataset to filter.
    - min_num_nodes: Minimum allowed number of nodes (inclusive, default: 1).
    - max_num_nodes: Maximum allowed number of nodes (inclusive, default: None, no upper limit).

    Returns:
    - filtered_dataset: The dataset after filtering.
    - stats: A dictionary containing statistics about the filtering.
    """
    label_list = [data.y for data in dataset]
    if max_label is not None:
        mask = [(n >= min_label) and (n <= max_label) for n in label_list]
    else:
        mask = [n >= min_label for n in label_list]
    filtered_dataset = [d for d, keep in zip(dataset, mask) if keep]

    stats = {
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "num_removed": len(dataset) - len(filtered_dataset),
        "min_label": min_label,
        "max_label": max_label,
    }
    return filtered_dataset, stats



def filter_dataset(dataset, min_num_nodes=1, max_num_nodes=None,
                   min_label=0.01, max_label=None, filter_indices=None):
    """
    Filters a dataset based on number of nodes and labels, and optionally filters out specific categorical features.

    Parameters:
    - dataset: The dataset to filter.
    - min_num_nodes: Minimum allowed number of nodes (inclusive, default: 1).
    - max_num_nodes: Maximum allowed number of nodes (inclusive, default: None, no upper limit).
    - min_label: Minimum allowed label value (inclusive, default: 0.01).
    - max_label: Maximum allowed label value (inclusive, default: None, no upper limit).
    - filter_indices: Indices of categorical features to filter out (default: None).

    Returns:
    - filtered_dataset: The dataset after filtering.
    - stats: A dictionary containing statistics about the filtering.
    """
    filtered_nodes_dataset, stats_nodes = filter_dataset_by_num_nodes(
        dataset, min_num_nodes, max_num_nodes)

    filtered_labels_dataset, stats_labels = filter_dataset_by_labels(
        dataset, min_label, max_label)

    if filter_indices is not None:
        filtered_material_dataset, stats_material = filter_categorical_features(dataset, filter_indices)

    # Find the set of graph names present in all filtered datasets
    names_nodes = set([g.graph_name for g in filtered_nodes_dataset])
    names_material = set([g.graph_name for g in filtered_material_dataset])
    names_labels = set([g.graph_name for g in filtered_labels_dataset])

    # Inner join: intersection of all sets
    # common_names = names_nodes
    common_names = names_nodes & names_material & names_labels

    # Create a dictionary for fast lookup by graph_name for each dataset
    dict = {g.graph_name: g for g in dataset}

    # Combine: only keep graphs present in all three filtered datasets
    filtered_dataset = [
        dict[name] for name in common_names
    ]

    stats = {
        "nodes": stats_nodes,
        "labels": stats_labels,
        "material": stats_material if filter_indices is not None else None,
        "final_size": len(filtered_dataset),
    }

    return filtered_dataset, stats
