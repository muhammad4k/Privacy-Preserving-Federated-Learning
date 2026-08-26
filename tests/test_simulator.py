"""Integration test for the Project Atlas federated simulator."""

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from evaluation.result_writer import save_simulation_result
from federated.simulator import FederatedSimulator


def main() -> None:
    """Run and validate a small federated learning simulation."""

    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    simulator = FederatedSimulator(
        dataset_name="mnist",
        client_subsets=client_subsets,
        test_dataset=test_dataset,
        num_rounds=3,
        learning_rate=0.01,
        momentum=0.9,
        local_epochs=1,
        batch_size=64,
    )

    result = simulator.run()

    csv_path, json_path = save_simulation_result(
        result=result,
        run_name="test_mnist_federated_simulation",
    )

    assert len(result.rounds) == 3
    assert result.rounds[-1].global_accuracy > 0.10
    assert csv_path.exists()
    assert json_path.exists()

    print("\nRound accuracies:")
    for round_result in result.rounds:
        print(
            f"Round {round_result.round_number}: "
            f"{round_result.global_accuracy:.4f}"
        )

    print("\nSaved result files:")
    print("CSV:", csv_path)
    print("JSON:", json_path)

    print("\nFederated simulator checks passed.")


if __name__ == "__main__":
    main()