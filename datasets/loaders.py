"""Dataset loading utilities for Project Atlas."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset
from torchvision import datasets, transforms


SUPPORTED_DATASETS = {"mnist", "fashion_mnist", "cifar10"}


def _mnist_transform() -> Callable:
    """Return the normalisation pipeline used for MNIST."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def _fashion_mnist_transform() -> Callable:
    """Return the normalisation pipeline used for Fashion-MNIST."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )


def _cifar10_train_transform() -> Callable:
    """Return the training transform used for CIFAR-10."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def _cifar10_test_transform() -> Callable:
    """Return the deterministic evaluation transform used for CIFAR-10."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def load_dataset(
    name: str,
    data_directory: str | Path = "data",
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    """
    Load a supported benchmark dataset.

    Args:
        name: Dataset identifier: mnist, fashion_mnist or cifar10.
        data_directory: Directory used to store downloaded datasets.
        download: Whether torchvision may download missing files.

    Returns:
        A tuple containing the training and test datasets.

    Raises:
        ValueError: If the requested dataset is unsupported.
    """

    dataset_name = name.strip().lower()
    root = Path(data_directory)

    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dataset '{name}'. Supported datasets: {supported}"
        )

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root=root,
            train=True,
            transform=_mnist_transform(),
            download=download,
        )
        test_dataset = datasets.MNIST(
            root=root,
            train=False,
            transform=_mnist_transform(),
            download=download,
        )

    elif dataset_name == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(
            root=root,
            train=True,
            transform=_fashion_mnist_transform(),
            download=download,
        )
        test_dataset = datasets.FashionMNIST(
            root=root,
            train=False,
            transform=_fashion_mnist_transform(),
            download=download,
        )

    else:
        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            transform=_cifar10_train_transform(),
            download=download,
        )
        test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            transform=_cifar10_test_transform(),
            download=download,
        )

    return train_dataset, test_dataset