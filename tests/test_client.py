"""Sanity test for one Project Atlas federated client."""

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from federated.client import ClientConfig, FederatedClient


def main() -> None:
    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    config = ClientConfig(
        client_id=0,
        learning_rate=0.01,
        momentum=0.9,
        local_epochs=1,
        batch_size=64,
    )

    client = FederatedClient(
        dataset_name="mnist",
        train_subset=client_subsets[0],
        test_dataset=test_dataset,
        config=config,
    )

    before = client.evaluate()
    print("Accuracy before local training:", round(before.accuracy, 4))

    parameters, num_examples, train_metrics = client.fit()

    after = client.evaluate()

    print("Local examples:", num_examples)
    print("Training loss:", round(train_metrics["train_loss"], 4))
    print(
        "Training runtime:",
        round(train_metrics["runtime_seconds"], 2),
        "seconds",
    )
    print("Accuracy after local training:", round(after.accuracy, 4))
    print("Returned parameter tensors:", len(parameters))

    assert num_examples == len(client_subsets[0])
    assert after.accuracy > before.accuracy
    assert len(parameters) > 0

    print("Federated client checks passed.")


if __name__ == "__main__":
    main()