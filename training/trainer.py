"""Local model training and evaluation utilities for Project Atlas."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingResult:
    """Results produced by local model training."""

    average_loss: float
    num_examples: int
    runtime_seconds: float


@dataclass
class EvaluationResult:
    """Classification metrics produced during model evaluation."""

    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    num_examples: int


def get_device() -> torch.device:
    """Select the best available compute device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """Create the SGD optimizer used across experiments."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    if not 0 <= momentum < 1:
        raise ValueError("momentum must be between 0 and 1")

    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    local_epochs: int,
    device: torch.device,
) -> TrainingResult:
    """Train a model locally for the requested number of epochs."""

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")

    if len(data_loader.dataset) == 0:
        raise ValueError("Training dataset must not be empty")

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    start_time = time.perf_counter()
    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    runtime_seconds = time.perf_counter() - start_time

    return TrainingResult(
        average_loss=total_loss / total_batches,
        num_examples=len(data_loader.dataset),
        runtime_seconds=runtime_seconds,
    )


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate a model using loss and classification metrics."""

    if len(data_loader.dataset) == 0:
        raise ValueError("Evaluation dataset must not be empty")

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * targets.size(0)
            total_correct += predictions.eq(targets).sum().item()

            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    num_examples = len(data_loader.dataset)

    return EvaluationResult(
        loss=total_loss / num_examples,
        accuracy=total_correct / num_examples,
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        num_examples=num_examples,
    )