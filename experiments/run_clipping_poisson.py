"""Federated ablation: per-sample clipping + Poisson sampling, no noise."""

from statistics import mean

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from evaluation.result_writer import save_simulation_result
from federated.aggregation import (
    get_model_parameters,
    set_model_parameters,
    weighted_fedavg,
)
from federated.simulator import RoundResult, SimulationResult
from models.cnn import create_model
from privacy.dp_engine import (
    make_training_clipping_poisson_no_noise,
)
from training.trainer import (
    create_optimizer,
    evaluate_model,
    get_device,
    train_model,
)
from torch.utils.data import DataLoader
from utils.reproducibility import set_global_seed


class ClippingPoissonClient:
    """Experimental client for clipping + Poisson sampling ablation."""

    def __init__(
        self,
        client_id,
        train_subset,
        test_dataset,
    ):
        self.client_id = client_id
        self.device = get_device()

        self.model = create_model("mnist")

        train_loader = DataLoader(
            train_subset,
            batch_size=64,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
        )

        optimizer = create_optimizer(
            model=self.model,
            learning_rate=0.01,
            momentum=0.9,
        )

        wrapped = make_training_clipping_poisson_no_noise(
            model=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=1.0,
        )

        self.model = wrapped.model
        self.optimizer = wrapped.optimizer
        self.train_loader = wrapped.data_loader

    def base_model(self):
        """Return underlying PyTorch model."""

        if hasattr(self.model, "_module"):
            return self.model._module

        return self.model

    def get_parameters(self):
        return get_model_parameters(
            self.base_model()
        )

    def set_parameters(self, parameters):
        set_model_parameters(
            self.base_model(),
            parameters,
        )

    def fit(self):
        # Match baseline/DP round behaviour:
        # reset momentum state each FL round.
        self.optimizer.state.clear()

        result = train_model(
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            local_epochs=1,
            device=self.device,
        )

        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            result,
        )


def main() -> None:
    set_global_seed(42)

    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    global_model = create_model("mnist")
    device = get_device()

    clients = [
        ClippingPoissonClient(
            client_id=i,
            train_subset=subset,
            test_dataset=test_dataset,
        )
        for i, subset in enumerate(client_subsets)
    ]

    result = SimulationResult(
        dataset_name="mnist",
        num_clients=5,
        num_rounds=3,
        privacy_enabled=False,
    )

    print("=" * 72)
    print("CLIPPING + POISSON SAMPLING ABLATION")
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

            client.set_parameters(global_parameters)

            (
                parameters,
                num_examples,
                train_result,
            ) = client.fit()

            parameter_sets.append(parameters)
            num_examples_list.append(num_examples)
            losses.append(train_result.average_loss)
            runtimes.append(train_result.runtime_seconds)

            print(
                f"  Client {client.client_id}: "
                f"loss={train_result.average_loss:.4f}, "
                f"runtime={train_result.runtime_seconds:.2f}s"
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
            f"  Global accuracy={global_metrics.accuracy:.4f}, "
            f"macro_f1={global_metrics.macro_f1:.4f}, "
            f"test_loss={global_metrics.loss:.4f}"
        )

    csv_path, json_path = save_simulation_result(
        result=result,
        run_name="clipping_poisson_no_noise_mnist",
    )

    print("\nAblation completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()