"""Reproducible client partitioning for federated experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset, Subset


def extract_targets(dataset: Dataset) -> np.ndarray:
    """
    Extract class labels from a torchvision-style dataset.

    Raises:
        AttributeError: If labels cannot be located.
    """

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError(
            "Dataset must expose labels using a 'targets' or 'labels' attribute."
        )

    if hasattr(targets, "numpy"):
        targets = targets.numpy()

    return np.asarray(targets, dtype=np.int64)


def validate_partition_request(
    dataset: Dataset,
    num_clients: int,
) -> None:
    """Validate common client-partitioning arguments."""

    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")

    if len(dataset) < num_clients:
        raise ValueError(
            "Dataset contains fewer samples than the requested number of clients"
        )


def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int,
) -> list[Subset]:
    """
    Partition a dataset randomly and approximately equally across clients.

    Every sample is assigned to exactly one client.
    """

    validate_partition_request(dataset, num_clients)

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(dataset))
    client_indices = np.array_split(shuffled_indices, num_clients)

    return [
        Subset(dataset, indices.tolist())
        for indices in client_indices
    ]


def partition_label_shards(
    dataset: Dataset,
    num_clients: int,
    shards_per_client: int,
    seed: int,
) -> list[Subset]:
    """
    Create a label-skewed non-IID partition using sorted label shards.

    Samples are sorted by class label, split into shards, shuffled, and
    allocated to clients. Fewer shards per client generally produce
    stronger label imbalance.

    Args:
        dataset: Training dataset to partition.
        num_clients: Number of federated clients.
        shards_per_client: Number of shards allocated to each client.
        seed: Random seed controlling shard allocation.
    """

    validate_partition_request(dataset, num_clients)

    if shards_per_client < 1:
        raise ValueError("shards_per_client must be at least 1")

    total_shards = num_clients * shards_per_client

    if len(dataset) < total_shards:
        raise ValueError(
            "Dataset is too small for the requested number of shards"
        )

    labels = extract_targets(dataset)
    sorted_indices = np.argsort(labels, kind="stable")
    shards = np.array_split(sorted_indices, total_shards)

    rng = np.random.default_rng(seed)
    shard_order = rng.permutation(total_shards)

    client_subsets: list[Subset] = []

    for client_id in range(num_clients):
        start = client_id * shards_per_client
        stop = start + shards_per_client
        selected_shards = shard_order[start:stop]

        indices = np.concatenate(
            [shards[shard_id] for shard_id in selected_shards]
        )

        rng.shuffle(indices)
        client_subsets.append(Subset(dataset, indices.tolist()))

    return client_subsets


def class_distribution(
    dataset: Dataset,
    subset: Subset,
) -> dict[int, int]:
    """Return the class-count distribution for one client subset."""

    labels = extract_targets(dataset)
    counts: defaultdict[int, int] = defaultdict(int)

    for index in subset.indices:
        counts[int(labels[index])] += 1

    return dict(sorted(counts.items()))