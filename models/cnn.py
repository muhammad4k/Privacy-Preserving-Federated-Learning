"""CNN model definitions for Project Atlas."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Compact CNN for MNIST, Fashion-MNIST, and CIFAR-10."""

    def __init__(self, input_channels: int, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        features = self.features(inputs)
        return self.classifier(features)


def create_model(dataset_name: str, num_classes: int = 10) -> nn.Module:
    """
    Create a CNN compatible with the selected dataset.

    Args:
        dataset_name: mnist, fashion_mnist, or cifar10.
        num_classes: Number of output classes.
    """

    name = dataset_name.strip().lower()

    if name in {"mnist", "fashion_mnist"}:
        input_channels = 1
    elif name == "cifar10":
        input_channels = 3
    else:
        raise ValueError(f"Unsupported dataset for model creation: {dataset_name}")

    return SmallCNN(
        input_channels=input_channels,
        num_classes=num_classes,
    )