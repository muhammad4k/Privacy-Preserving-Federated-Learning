"""End-to-end federated learning simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from torch.utils.data import Dataset, Subset

from federated.aggregation import (
    get_model_parameters,
    set_model_parameters,
    weighted_fedavg,
)
from federated.client import ClientConfig, FederatedClient
from models.cnn import create_model
from privacy.dp_engine import DifferentialPrivacyConfig
from training.trainer import EvaluationResult, evaluate_model, get_device


@dataclass
class RoundResult:
    """Metrics recorded after one federated learning round."""

    round_number: int

    global_loss: float
    global_accuracy: float
    global_macro_precision: float
    global_macro_recall: float
    global_macro_f1: float

    mean_client_train_loss: float
    mean_client_runtime_seconds: float

    # Privacy metrics are None for baseline experiments.
    mean_client_epsilon: float | None = None
    max_client_epsilon: float | None = None
    mean_noise_multiplier: float | None = None


@dataclass
class SimulationResult:
    """Complete history and metadata for one federated simulation."""

    dataset_name: str
    num_clients: int
    num_rounds: int

    privacy_enabled: bool = False
    target_epsilon: float | None = None
    target_delta: float | None = None
    max_grad_norm: float | None = None

    rounds: list[RoundResult] = field(default_factory=list)


class FederatedSimulator:
    """
    Coordinate multiple local clients using weighted FedAvg.

    Differential Privacy is optional. When enabled, every client
    maintains its own persistent privacy accountant throughout the
    complete federated simulation.
    """

    def __init__(
        self,
        dataset_name: str,
        client_subsets: list[Subset],
        test_dataset: Dataset,
        num_rounds: int,
        learning_rate: float,
        momentum: float,
        local_epochs: int,
        batch_size: int,
        test_batch_size: int = 256,
        dp_config: DifferentialPrivacyConfig | None = None,
    ) -> None:

        if len(client_subsets) < 2:
            raise ValueError(
                "At least two clients are required"
            )

        if num_rounds < 1:
            raise ValueError(
                "num_rounds must be at least 1"
            )

        if local_epochs < 1:
            raise ValueError(
                "local_epochs must be at least 1"
            )

        self.dataset_name = dataset_name
        self.client_subsets = client_subsets
        self.test_dataset = test_dataset

        self.num_rounds = num_rounds
        self.local_epochs = local_epochs

        self.dp_config = dp_config
        self.device = get_device()

        self.global_model = create_model(
            dataset_name
        )

        # Privacy budget is calibrated against the COMPLETE
        # intended private training horizon.
        total_private_epochs = (
            num_rounds * local_epochs
            if dp_config is not None
            else None
        )

        self.clients = [
            FederatedClient(
                dataset_name=dataset_name,
                train_subset=subset,
                test_dataset=test_dataset,
                config=ClientConfig(
                    client_id=client_id,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    test_batch_size=test_batch_size,
                ),
                dp_config=dp_config,
                total_private_epochs=total_private_epochs,
            )
            for client_id, subset
            in enumerate(client_subsets)
        ]

    @property
    def privacy_enabled(self) -> bool:
        """Return whether Differential Privacy is enabled."""

        return self.dp_config is not None

    def evaluate_global_model(
        self,
    ) -> EvaluationResult:
        """Evaluate the global model on the shared test set."""

        test_loader = self.clients[0].test_loader

        return evaluate_model(
            model=self.global_model,
            data_loader=test_loader,
            device=self.device,
        )

    def run(self) -> SimulationResult:
        """Execute the configured federated learning simulation."""

        result = SimulationResult(
            dataset_name=self.dataset_name,
            num_clients=len(self.clients),
            num_rounds=self.num_rounds,
            privacy_enabled=self.privacy_enabled,
            target_epsilon=(
                self.dp_config.target_epsilon
                if self.dp_config
                else None
            ),
            target_delta=(
                self.dp_config.target_delta
                if self.dp_config
                else None
            ),
            max_grad_norm=(
                self.dp_config.max_grad_norm
                if self.dp_config
                else None
            ),
        )

        initial_metrics = (
            self.evaluate_global_model()
        )

        print("=" * 72)
        print(
            "FEDERATED LEARNING SIMULATION"
        )
        print("=" * 72)

        print(
            f"Dataset: {self.dataset_name}"
        )
        print(
            f"Clients: {len(self.clients)}"
        )
        print(
            f"Rounds: {self.num_rounds}"
        )
        print(
            f"Local epochs: {self.local_epochs}"
        )
        print(
            f"Device: {self.device}"
        )
        print(
            f"Privacy enabled: "
            f"{self.privacy_enabled}"
        )

        if self.privacy_enabled:
            print(
                f"Target epsilon: "
                f"{self.dp_config.target_epsilon}"
            )
            print(
                f"Delta: "
                f"{self.dp_config.target_delta}"
            )
            print(
                f"Max grad norm: "
                f"{self.dp_config.max_grad_norm}"
            )

        print(
            f"Initial global accuracy: "
            f"{initial_metrics.accuracy:.4f}"
        )

        print("=" * 72)

        for round_number in range(
            1,
            self.num_rounds + 1,
        ):

            global_parameters = (
                get_model_parameters(
                    self.global_model
                )
            )

            client_parameter_sets = []
            client_num_examples = []

            client_train_losses = []
            client_runtimes = []

            client_epsilons = []
            client_noise_multipliers = []

            print(
                f"\nRound "
                f"{round_number}/{self.num_rounds}"
            )

            for client in self.clients:

                # Broadcast current global parameters.
                client.set_parameters(
                    global_parameters
                )

                (
                    parameters,
                    num_examples,
                    metrics,
                ) = client.fit()

                client_parameter_sets.append(
                    parameters
                )

                client_num_examples.append(
                    num_examples
                )

                client_train_losses.append(
                    metrics["train_loss"]
                )

                client_runtimes.append(
                    metrics["runtime_seconds"]
                )

                message = (
                    f"  Client "
                    f"{client.config.client_id}: "
                    f"examples={num_examples}, "
                    f"loss="
                    f"{metrics['train_loss']:.4f}, "
                    f"runtime="
                    f"{metrics['runtime_seconds']:.2f}s"
                )

                if self.privacy_enabled:

                    epsilon = metrics["epsilon"]
                    noise = metrics[
                        "noise_multiplier"
                    ]

                    client_epsilons.append(
                        epsilon
                    )

                    client_noise_multipliers.append(
                        noise
                    )

                    message += (
                        f", epsilon={epsilon:.4f}, "
                        f"noise={noise:.4f}"
                    )

                print(message)

            aggregated_parameters = (
                weighted_fedavg(
                    client_parameters=
                        client_parameter_sets,
                    client_num_examples=
                        client_num_examples,
                )
            )

            set_model_parameters(
                self.global_model,
                aggregated_parameters,
            )

            global_metrics = (
                self.evaluate_global_model()
            )

            mean_epsilon = None
            max_epsilon = None
            mean_noise = None

            if self.privacy_enabled:
                mean_epsilon = mean(
                    client_epsilons
                )

                max_epsilon = max(
                    client_epsilons
                )

                mean_noise = mean(
                    client_noise_multipliers
                )

            round_result = RoundResult(
                round_number=round_number,

                global_loss=
                    global_metrics.loss,

                global_accuracy=
                    global_metrics.accuracy,

                global_macro_precision=
                    global_metrics.macro_precision,

                global_macro_recall=
                    global_metrics.macro_recall,

                global_macro_f1=
                    global_metrics.macro_f1,

                mean_client_train_loss=
                    mean(client_train_losses),

                mean_client_runtime_seconds=
                    mean(client_runtimes),

                mean_client_epsilon=
                    mean_epsilon,

                max_client_epsilon=
                    max_epsilon,

                mean_noise_multiplier=
                    mean_noise,
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

            if self.privacy_enabled:
                print(
                    f"  Privacy: "
                    f"mean epsilon="
                    f"{mean_epsilon:.4f}, "
                    f"max epsilon="
                    f"{max_epsilon:.4f}"
                )

        print(
            "\nSimulation completed."
        )

        if self.privacy_enabled:

            final_epsilons = [
                client.get_epsilon()
                for client in self.clients
            ]

            print(
                "Final client epsilons:"
            )

            for client, epsilon in zip(
                self.clients,
                final_epsilons,
            ):
                print(
                    f"  Client "
                    f"{client.config.client_id}: "
                    f"{epsilon:.4f}"
                )

        return result