"""Sanity checks for Differential Privacy integration."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from datasets.loaders import load_dataset
from models.cnn import create_model
from privacy.dp_engine import (
    DifferentialPrivacyConfig,
    get_spent_epsilon,
    make_training_private,
)
from training.trainer import create_optimizer


def main() -> None:
    train_dataset, _ = load_dataset("mnist")

    # Small subset because this is a DP integration test,
    # not a performance experiment.
    subset = Subset(
        train_dataset,
        list(range(2048)),
    )

    train_loader = DataLoader(
        subset,
        batch_size=64,
        shuffle=True,
    )

    model = create_model("mnist")

    optimizer = create_optimizer(
        model=model,
        learning_rate=0.01,
        momentum=0.9,
    )

    config = DifferentialPrivacyConfig(
        target_epsilon=5.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    private_objects = make_training_private(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        config=config,
        total_epochs=3,
    )

    print(
        "Noise multiplier:",
        round(private_objects.noise_multiplier, 4),
    )

    initial_epsilon = get_spent_epsilon(
        private_objects.privacy_engine,
        delta=config.target_delta,
    )

    print(
        "Initial epsilon spent:",
        round(initial_epsilon, 4),
    )

    assert initial_epsilon == 0.0
    assert private_objects.noise_multiplier > 0

    # Perform one private training epoch.
    private_objects.model.train()

    criterion = nn.CrossEntropyLoss()

    for inputs, targets in private_objects.data_loader:
        private_objects.optimizer.zero_grad()

        outputs = private_objects.model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        private_objects.optimizer.step()

    epsilon_after_epoch = get_spent_epsilon(
        private_objects.privacy_engine,
        delta=config.target_delta,
    )

    print(
        "Epsilon after one private epoch:",
        round(epsilon_after_epoch, 4),
    )

    assert epsilon_after_epoch > 0.0

    print(
        "Differential Privacy engine checks passed."
    )


if __name__ == "__main__":
    main()