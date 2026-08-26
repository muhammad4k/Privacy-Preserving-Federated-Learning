"""Full federated clipping-only ablation experiment."""

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from evaluation.result_writer import save_simulation_result
from federated.aggregation import (
    get_model_parameters,
    set_model_parameters,
    weighted_fedavg,
)
from federated.client import ClientConfig, FederatedClient
from models.cnn import create_model
from statistics import mean
from training.trainer import evaluate_model, get_device
from utils.reproducibility import set_global_seed

from federated.simulator import RoundResult, SimulationResult


def main() -> None:

    set_global_seed(42)

    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    device = get_device()
    global_model = create_model("mnist")

    clients = [
        FederatedClient(
            dataset_name="mnist",
            train_subset=subset,
            test_dataset=test_dataset,
            config=ClientConfig(
                client_id=client_id,
                learning_rate=0.01,
                momentum=0.9,
                local_epochs=1,
                batch_size=64,
            ),
            clipping_only=True,
            clipping_max_grad_norm=1.0,
        )
        for client_id, subset in enumerate(client_subsets)
    ]

    result = SimulationResult(
        dataset_name="mnist",
        num_clients=5,
        num_rounds=3,
        privacy_enabled=False,
    )

    print("=" * 72)
    print("CLIPPING-ONLY FEDERATED ABLATION")
    print("=" * 72)

    for round_number in range(1, 4):

        print(f"\nRound {round_number}/3")

        global_parameters = get_model_parameters(
            global_model
        )

        parameter_sets = []
        num_examples_list = []
        losses = []
        runtimes = []

        for client in clients:

            client.set_parameters(
                global_parameters
            )

            parameters, num_examples, metrics = client.fit()

            parameter_sets.append(parameters)
            num_examples_list.append(num_examples)

            losses.append(metrics["train_loss"])
            runtimes.append(metrics["runtime_seconds"])

            print(
                f"  Client {client.config.client_id}: "
                f"loss={metrics['train_loss']:.4f}, "
                f"runtime={metrics['runtime_seconds']:.2f}s"
            )

        averaged_parameters = weighted_fedavg(
            client_parameters=parameter_sets,
            client_num_examples=num_examples_list,
        )

        set_model_parameters(
            global_model,
            averaged_parameters,
        )

        global_metrics = evaluate_model(
            model=global_model,
            data_loader=clients[0].test_loader,
            device=device,
        )

        result.rounds.append(
            RoundResult(
                round_number=round_number,
                global_loss=global_metrics.loss,
                global_accuracy=global_metrics.accuracy,
                global_macro_precision=global_metrics.macro_precision,
                global_macro_recall=global_metrics.macro_recall,
                global_macro_f1=global_metrics.macro_f1,
                mean_client_train_loss=mean(losses),
                mean_client_runtime_seconds=mean(runtimes),
            )
        )

        print(
            f"  Global accuracy="
            f"{global_metrics.accuracy:.4f}, "
            f"macro_f1="
            f"{global_metrics.macro_f1:.4f}, "
            f"test_loss="
            f"{global_metrics.loss:.4f}"
        )

    csv_path, json_path = save_simulation_result(
        result=result,
        run_name="clipping_only_mnist",
    )

    print("\nClipping-only experiment completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()