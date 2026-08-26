"""Sanity checks for Project Atlas dataset partitioning."""

from datasets.loaders import load_dataset
from datasets.partitioning import (
    class_distribution,
    partition_iid,
    partition_label_shards,
)


def main() -> None:
    train_dataset, test_dataset = load_dataset("mnist")

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    iid_clients = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    print("\nIID client sizes:")
    for client_id, subset in enumerate(iid_clients):
        distribution = class_distribution(train_dataset, subset)
        print(
            f"Client {client_id}: "
            f"{len(subset)} samples | classes={distribution}"
        )

    non_iid_clients = partition_label_shards(
        dataset=train_dataset,
        num_clients=5,
        shards_per_client=2,
        seed=42,
    )

    print("\nNon-IID client sizes:")
    for client_id, subset in enumerate(non_iid_clients):
        distribution = class_distribution(train_dataset, subset)
        print(
            f"Client {client_id}: "
            f"{len(subset)} samples | classes={distribution}"
        )

    assert sum(len(client) for client in iid_clients) == len(train_dataset)
    assert sum(len(client) for client in non_iid_clients) == len(train_dataset)

    print("\nPartitioning checks passed.")


if __name__ == "__main__":
    main()