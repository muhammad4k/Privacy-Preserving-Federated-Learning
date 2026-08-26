"""Federated client implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opacus.grad_sample import GradSampleModule
from torch.utils.data import DataLoader, Dataset, Subset

from federated.aggregation import (
    StateDictionary,
    get_model_parameters,
    set_model_parameters,
)
from models.cnn import create_model
from privacy.dp_engine import (
    DifferentialPrivacyConfig,
    get_spent_epsilon,
    make_training_clipping_only,
    make_training_private,
)
from training.trainer import (
    EvaluationResult,
    create_optimizer,
    evaluate_model,
    get_device,
    train_model,
)


@dataclass(frozen=True)
class ClientConfig:
    """Configuration for one simulated federated client."""

    client_id: int
    learning_rate: float
    momentum: float
    local_epochs: int
    batch_size: int
    test_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be between 0 and 1")

        if self.local_epochs < 1:
            raise ValueError("local_epochs must be at least 1")

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if self.test_batch_size < 1:
            raise ValueError("test_batch_size must be at least 1")


class FederatedClient:
    """Represent one simulated federated learning client."""

    def __init__(
        self,
        dataset_name: str,
        train_subset: Subset,
        test_dataset: Dataset,
        config: ClientConfig,
        dp_config: DifferentialPrivacyConfig | None = None,
        total_private_epochs: int | None = None,
        clipping_only: bool = False,
        clipping_max_grad_norm: float = 1.0,
    ) -> None:

        if len(train_subset) == 0:
            raise ValueError("train_subset must not be empty")

        if len(test_dataset) == 0:
            raise ValueError("test_dataset must not be empty")

        if dp_config is not None and clipping_only:
            raise ValueError(
                "DP and clipping-only modes cannot be enabled together"
            )

        self.dataset_name = dataset_name
        self.config = config
        self.dp_config = dp_config
        self.clipping_only = clipping_only
        self.device = get_device()

        self.model = create_model(dataset_name)

        self.train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
        )

        self.optimizer = None
        self.privacy_engine = None
        self.noise_multiplier = None

        if self.dp_config is not None:

            if total_private_epochs is None:
                raise ValueError(
                    "total_private_epochs is required when DP is enabled"
                )

            optimizer = create_optimizer(
                model=self.model,
                learning_rate=config.learning_rate,
                momentum=config.momentum,
            )

            private_objects = make_training_private(
                model=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                config=self.dp_config,
                total_epochs=total_private_epochs,
            )

            self.model = private_objects.model
            self.optimizer = private_objects.optimizer
            self.train_loader = private_objects.data_loader
            self.privacy_engine = private_objects.privacy_engine
            self.noise_multiplier = private_objects.noise_multiplier

        elif self.clipping_only:

            optimizer = create_optimizer(
                model=self.model,
                learning_rate=config.learning_rate,
                momentum=config.momentum,
            )

            clipping_objects = make_training_clipping_only(
                model=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                max_grad_norm=clipping_max_grad_norm,
            )

            self.model = clipping_objects.model
            self.optimizer = clipping_objects.optimizer
            self.train_loader = clipping_objects.data_loader
            self.privacy_engine = clipping_objects.privacy_engine
            self.noise_multiplier = 0.0

    @property
    def privacy_enabled(self) -> bool:
        return self.dp_config is not None

    def _parameter_model(self):
        if isinstance(self.model, GradSampleModule):
            return self.model._module

        return self.model

    def get_parameters(self) -> StateDictionary:
        return get_model_parameters(
            self._parameter_model()
        )

    def set_parameters(
        self,
        parameters: StateDictionary,
    ) -> None:
        set_model_parameters(
            self._parameter_model(),
            parameters,
        )

    def get_epsilon(self) -> float | None:

        if not self.privacy_enabled:
            return None

        if self.privacy_engine is None:
            raise RuntimeError(
                "Privacy engine has not been initialised"
            )

        return get_spent_epsilon(
            privacy_engine=self.privacy_engine,
            delta=self.dp_config.target_delta,
        )

    def fit(
        self,
    ) -> tuple[StateDictionary, int, dict[str, float]]:

        if self.privacy_enabled or self.clipping_only:

            if self.optimizer is None:
                raise RuntimeError(
                    "Wrapped optimizer has not been initialised"
                )

            # Remove momentum accumulated during the previous FL round
            # while preserving the Opacus wrapper/accountant.
            self.optimizer.state.clear()

            optimizer = self.optimizer

        else:

            optimizer = create_optimizer(
                model=self.model,
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum,
            )

        result = train_model(
            model=self.model,
            data_loader=self.train_loader,
            optimizer=optimizer,
            local_epochs=self.config.local_epochs,
            device=self.device,
        )

        metrics = {
            "train_loss": float(result.average_loss),
            "runtime_seconds": float(result.runtime_seconds),
        }

        epsilon = self.get_epsilon()

        if epsilon is not None:
            metrics["epsilon"] = float(epsilon)
            metrics["noise_multiplier"] = float(
                self.noise_multiplier
            )

        return (
            self.get_parameters(),
            result.num_examples,
            metrics,
        )

    def evaluate(self) -> EvaluationResult:

        return evaluate_model(
            model=self.model,
            data_loader=self.test_loader,
            device=self.device,
        )

    def summary(self) -> dict[str, Any]:

        return {
            "client_id": self.config.client_id,
            "dataset_name": self.dataset_name,
            "num_train_examples": len(self.train_loader.dataset),
            "num_test_examples": len(self.test_loader.dataset),
            "device": str(self.device),
            "privacy_enabled": self.privacy_enabled,
            "clipping_only": self.clipping_only,
        }