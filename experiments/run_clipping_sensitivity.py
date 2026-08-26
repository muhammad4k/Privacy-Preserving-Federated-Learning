"""Clipping sensitivity study for the federated learning framework."""

from statistics import mean

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from evaluation.result_writer import save_simulation_result
from federated.aggregation import (
    get_model_parameters,
    set_model_parameters,
    weighted_fedavg,
)
from federated.client import ClientConfig, FederatedClient
from federated.simulator import RoundResult, SimulationResult
from models.cnn import create_model
from training.trainer import evaluate_model, get_device
from utils.reproducibility import set_global_seed


CLIPPING_NORMS = [0.5, 1.0, 2.0, 5.0]

NUM_CLIENTS = 5
NUM_ROUNDS = 15
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SEED = 42


def run_clipping_experiment(
    max_grad_norm: float,
) -> SimulationResult:
    """Run one clipping-only federated experiment."""

    set_global_seed(SEED)

    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=NUM_CLIENTS,
        seed=SEED,
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
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
                local_epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
            ),
            clipping_only=True,
            clipping_max_grad_norm=max_grad_norm,
        )
        for client_id, subset in enumerate(client_subsets)
    ]

    result = SimulationResult(
        dataset_name="mnist",
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        privacy_enabled=False,
    )

    print("=" * 72)
    print(
        f"CLIPPING SENSITIVITY EXPERIMENT "
        f"(max_grad_norm={max_grad_norm})"
    )
    print("=" * 72)

    for round_number in range(1, NUM_ROUNDS + 1):

        print(
            f"\nRound {round_number}/{NUM_ROUNDS}"
        )

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

            (
                parameters,
                num_examples,
                metrics,
            ) = client.fit()

            parameter_sets.append(
                parameters
            )

            num_examples_list.append(
                num_examples
            )

            losses.append(
                metrics["train_loss"]
            )

            runtimes.append(
                metrics["runtime_seconds"]
            )

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

        round_result = RoundResult(
            round_number=round_number,
            global_loss=global_metrics.loss,
            global_accuracy=global_metrics.accuracy,
            global_macro_precision=
                global_metrics.macro_precision,
            global_macro_recall=
                global_metrics.macro_recall,
            global_macro_f1=
                global_metrics.macro_f1,
            mean_client_train_loss=
                mean(losses),
            mean_client_runtime_seconds=
                mean(runtimes),
        )

        result.rounds.append(
            round_result
        )

        print(
            f"  Global accuracy="
            f"{global_metrics.accuracy:.4f}, "
            f"macro_f1="
            f"{global_metrics.macro_f1:.4f}, "
            f"test_loss="
            f"{global_metrics.loss:.4f}"
        )

    run_name = (
        "clipping_only_mnist_"
        f"c{str(max_grad_norm).replace('.', '_')}"
    )

    csv_path, json_path = save_simulation_result(
        result=result,
        run_name=run_name,
    )

    final_round = result.rounds[-1]

    print("\nFinal result")
    print(
        f"Clipping norm: {max_grad_norm}"
    )
    print(
        f"Accuracy: "
        f"{final_round.global_accuracy:.4f}"
    )
    print(
        f"Macro-F1: "
        f"{final_round.global_macro_f1:.4f}"
    )
    print(
        f"Test loss: "
        f"{final_round.global_loss:.4f}"
    )
    print("CSV:", csv_path)
    print("JSON:", json_path)

    return result


def main() -> None:
    """Run the complete clipping sensitivity study."""

    summary = []

    for clipping_norm in CLIPPING_NORMS:

        result = run_clipping_experiment(
            clipping_norm
        )

        final_round = result.rounds[-1]

        summary.append(
            (
                clipping_norm,
                final_round.global_accuracy,
                final_round.global_macro_f1,
                final_round.global_loss,
            )
        )

    print("\n" + "=" * 72)
    print("CLIPPING SENSITIVITY SUMMARY")
    print("=" * 72)

    for (
        clipping_norm,
        accuracy,
        macro_f1,
        loss,
    ) in summary:

        print(
            f"C={clipping_norm:<4} | "
            f"accuracy={accuracy:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"loss={loss:.4f}"
        )


if __name__ == "__main__":
    main()