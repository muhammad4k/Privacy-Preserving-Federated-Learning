"""Configuration-driven experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.loaders import load_dataset
from datasets.partitioning import (
    partition_iid,
    partition_label_shards,
)
from evaluation.result_writer import (
    save_simulation_result,
)
from federated.simulator import (
    FederatedSimulator,
)
from privacy.dp_engine import (
    DifferentialPrivacyConfig,
)
from utils.config import Config
from utils.reproducibility import (
    set_global_seed,
)
from visualisation.plots import (
    plot_simulation_metrics,
)


def build_client_partitions(
    train_dataset: Any,
    partition_type: str,
    num_clients: int,
    seed: int,
):
    """Create client partitions from configuration."""

    partition_name = (
        partition_type.strip().lower()
    )

    if partition_name == "iid":
        return partition_iid(
            dataset=train_dataset,
            num_clients=num_clients,
            seed=seed,
        )

    if partition_name == "non_iid":
        return partition_label_shards(
            dataset=train_dataset,
            num_clients=num_clients,
            shards_per_client=2,
            seed=seed,
        )

    raise ValueError(
        "Unsupported partition type. "
        "Use 'iid' or 'non_iid'."
    )


def run_experiment(
    config_path: str | Path =
        "configs/default.yaml",
):
    """
    Run one complete federated experiment
    from a YAML configuration file.
    """

    config = Config(
        config_path
    )

    experiment_name = str(
        config.get(
            "experiment",
            "name",
        )
    )

    seed = int(
        config.get(
            "experiment",
            "random_seed",
        )
    )

    dataset_name = str(
        config.get(
            "dataset",
            "name",
        )
    )

    partition_type = str(
        config.get(
            "partition",
            "type",
        )
    )

    num_clients = int(
        config.get(
            "partition",
            "num_clients",
        )
    )

    num_rounds = int(
        config.get(
            "training",
            "num_rounds",
        )
    )

    local_epochs = int(
        config.get(
            "training",
            "local_epochs",
        )
    )

    batch_size = int(
        config.get(
            "training",
            "batch_size",
        )
    )

    learning_rate = float(
        config.get(
            "training",
            "learning_rate",
        )
    )

    momentum = float(
        config.get(
            "training",
            "momentum",
        )
    )

    privacy_enabled = bool(
        config.get(
            "privacy",
            "enabled",
        )
    )

    epsilon = float(
        config.get(
            "privacy",
            "epsilon",
        )
    )

    delta = float(
        config.get(
            "privacy",
            "delta",
        )
    )

    max_grad_norm = float(
        config.get(
            "privacy",
            "max_grad_norm",
        )
    )

    save_csv = bool(
        config.get(
            "results",
            "save_csv",
        )
    )

    save_json = bool(
        config.get(
            "results",
            "save_json",
        )
    )

    save_plots = bool(
        config.get(
            "results",
            "save_plots",
        )
    )

    set_global_seed(
        seed
    )

    dp_config = None

    if privacy_enabled:
        dp_config = (
            DifferentialPrivacyConfig(
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=max_grad_norm,
            )
        )

    print("=" * 72)
    print(
        "CONFIGURATION-DRIVEN EXPERIMENT"
    )
    print("=" * 72)

    print(
        f"Experiment: {experiment_name}"
    )
    print(
        f"Dataset: {dataset_name}"
    )
    print(
        f"Partition: {partition_type}"
    )
    print(
        f"Clients: {num_clients}"
    )
    print(
        f"Rounds: {num_rounds}"
    )
    print(
        f"Local epochs: {local_epochs}"
    )
    print(
        f"Batch size: {batch_size}"
    )
    print(
        f"Learning rate: {learning_rate}"
    )
    print(
        f"Momentum: {momentum}"
    )
    print(
        f"Seed: {seed}"
    )
    print(
        f"Privacy enabled: "
        f"{privacy_enabled}"
    )

    if privacy_enabled:
        print(
            f"Target epsilon: {epsilon}"
        )
        print(
            f"Delta: {delta}"
        )
        print(
            f"Max grad norm: "
            f"{max_grad_norm}"
        )

    print("=" * 72)

    train_dataset, test_dataset = (
        load_dataset(
            name=dataset_name,
            data_directory="data",
            download=True,
        )
    )

    client_subsets = (
        build_client_partitions(
            train_dataset=
                train_dataset,
            partition_type=
                partition_type,
            num_clients=
                num_clients,
            seed=
                seed,
        )
    )

    simulator = FederatedSimulator(
        dataset_name=dataset_name,
        client_subsets=client_subsets,
        test_dataset=test_dataset,

        num_rounds=num_rounds,

        learning_rate=learning_rate,
        momentum=momentum,

        local_epochs=local_epochs,
        batch_size=batch_size,

        dp_config=dp_config,
    )

    result = simulator.run()

    csv_path = None
    json_path = None

    if save_csv or save_json:

        csv_path, json_path = (
            save_simulation_result(
                result=result,
                run_name=
                    experiment_name,
            )
        )

    print(
        "\nExperiment outputs:"
    )

    if (
        save_csv
        and csv_path is not None
    ):
        print(
            "CSV:",
            csv_path,
        )

    if (
        save_json
        and json_path is not None
    ):
        print(
            "JSON:",
            json_path,
        )

    if save_plots:

        if csv_path is None:
            raise RuntimeError(
                "Plot generation requires "
                "CSV output. Set "
                "results.save_csv to true."
            )

        (
            accuracy_path,
            loss_path,
            f1_path,
        ) = plot_simulation_metrics(
            csv_path=csv_path,
            run_name=
                experiment_name,
        )

        print(
            "Accuracy plot:",
            accuracy_path,
        )

        print(
            "Loss plot:",
            loss_path,
        )

        print(
            "Macro-F1 plot:",
            f1_path,
        )

    return result


if __name__ == "__main__":
    run_experiment()