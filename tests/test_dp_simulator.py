"""Integration test for the DP federated simulator."""

from torch.utils.data import Subset

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from federated.simulator import FederatedSimulator
from privacy.dp_engine import DifferentialPrivacyConfig
from utils.reproducibility import set_global_seed


def main() -> None:

    set_global_seed(42)

    train_dataset, test_dataset = (
        load_dataset("mnist")
    )

    # Small subset keeps this integration test quick.
    small_train_dataset = Subset(
        train_dataset,
        list(range(6000)),
    )

    client_subsets = partition_iid(
        dataset=small_train_dataset,
        num_clients=3,
        seed=42,
    )

    dp_config = (
        DifferentialPrivacyConfig(
            target_epsilon=5.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )
    )

    simulator = FederatedSimulator(
        dataset_name="mnist",
        client_subsets=client_subsets,
        test_dataset=test_dataset,

        num_rounds=2,

        learning_rate=0.01,
        momentum=0.9,

        local_epochs=1,
        batch_size=64,

        dp_config=dp_config,
    )

    result = simulator.run()

    assert len(result.rounds) == 2

    round_1 = result.rounds[0]
    round_2 = result.rounds[1]

    assert (
        round_1.mean_client_epsilon
        is not None
    )

    assert (
        round_2.mean_client_epsilon
        is not None
    )

    assert (
        round_2.mean_client_epsilon
        >
        round_1.mean_client_epsilon
    )

    assert (
        round_2.max_client_epsilon
        <= 5.5
    )

    print(
        "\nRound 1 mean epsilon:",
        round(
            round_1.mean_client_epsilon,
            4,
        ),
    )

    print(
        "Round 2 mean epsilon:",
        round(
            round_2.mean_client_epsilon,
            4,
        ),
    )

    print(
        "DP federated simulator "
        "checks passed."
    )


if __name__ == "__main__":
    main()