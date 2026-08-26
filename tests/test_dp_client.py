"""Integration test for a differentially private federated client."""

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from federated.client import (
    ClientConfig,
    FederatedClient,
)
from privacy.dp_engine import DifferentialPrivacyConfig
from utils.reproducibility import set_global_seed


def main() -> None:
    set_global_seed(42)

    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    client_config = ClientConfig(
        client_id=0,
        learning_rate=0.01,
        momentum=0.9,
        local_epochs=1,
        batch_size=64,
    )

    dp_config = DifferentialPrivacyConfig(
        target_epsilon=5.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    client = FederatedClient(
        dataset_name="mnist",
        train_subset=client_subsets[0],
        test_dataset=test_dataset,
        config=client_config,
        dp_config=dp_config,

        # Simulates three FL rounds with one local epoch per round.
        total_private_epochs=3,
    )

    print("Privacy enabled:", client.privacy_enabled)
    print(
        "Noise multiplier:",
        round(client.noise_multiplier, 4),
    )

    initial_epsilon = client.get_epsilon()

    print(
        "Initial epsilon:",
        round(initial_epsilon, 4),
    )

    assert initial_epsilon == 0.0

    # Federated round 1
    _, num_examples_1, metrics_1 = client.fit()

    epsilon_round_1 = metrics_1["epsilon"]

    print("\nAfter round 1")
    print("Examples:", num_examples_1)
    print(
        "Train loss:",
        round(metrics_1["train_loss"], 4),
    )
    print(
        "Epsilon:",
        round(epsilon_round_1, 4),
    )

    # Federated round 2
    _, num_examples_2, metrics_2 = client.fit()

    epsilon_round_2 = metrics_2["epsilon"]

    print("\nAfter round 2")
    print("Examples:", num_examples_2)
    print(
        "Train loss:",
        round(metrics_2["train_loss"], 4),
    )
    print(
        "Epsilon:",
        round(epsilon_round_2, 4),
    )

    # Federated round 3
    parameters, num_examples_3, metrics_3 = client.fit()

    epsilon_round_3 = metrics_3["epsilon"]

    print("\nAfter round 3")
    print("Examples:", num_examples_3)
    print(
        "Train loss:",
        round(metrics_3["train_loss"], 4),
    )
    print(
        "Epsilon:",
        round(epsilon_round_3, 4),
    )

    assert num_examples_1 == len(client_subsets[0])
    assert num_examples_2 == len(client_subsets[0])
    assert num_examples_3 == len(client_subsets[0])

    assert epsilon_round_1 > 0.0
    assert epsilon_round_2 > epsilon_round_1
    assert epsilon_round_3 > epsilon_round_2

    # make_private_with_epsilon was configured for the full
    # three-epoch private training duration, so final epsilon
    # should be close to the configured target rather than
    # epsilon=5 being restarted every round.
    assert epsilon_round_3 <= 5.5

    assert len(parameters) > 0

    print("\nReturned parameter tensors:", len(parameters))
    print(
        "Final cumulative epsilon:",
        round(epsilon_round_3, 4),
    )
    print(
        "Differentially private federated client checks passed."
    )


if __name__ == "__main__":
    main()