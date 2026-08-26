"""Differential Privacy utilities for the federated learning framework."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from opacus import PrivacyEngine
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    """Configuration for local Differential Privacy."""

    target_epsilon: float
    target_delta: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        if self.target_epsilon <= 0:
            raise ValueError("target_epsilon must be greater than zero")

        if not 0 < self.target_delta < 1:
            raise ValueError(
                "target_delta must be strictly between zero and one"
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be greater than zero"
            )


@dataclass
class PrivateTrainingObjects:
    """Objects returned after enabling Differential Privacy."""

    model: nn.Module
    optimizer: Optimizer
    data_loader: DataLoader
    privacy_engine: PrivacyEngine
    noise_multiplier: float


def make_training_private(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    config: DifferentialPrivacyConfig,
    total_epochs: int,
) -> PrivateTrainingObjects:
    """
    Make a local training process differentially private.

    The target privacy budget applies across the complete number of
    epochs supplied through ``total_epochs``.
    """

    if total_epochs < 1:
        raise ValueError("total_epochs must be at least 1")

    privacy_engine = PrivacyEngine(
        accountant="prv"
    )

    private_model, private_optimizer, private_loader = (
        privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
            epochs=total_epochs,
            max_grad_norm=config.max_grad_norm,
        )
    )

    noise_multiplier = float(
        private_optimizer.noise_multiplier
    )

    return PrivateTrainingObjects(
        model=private_model,
        optimizer=private_optimizer,
        data_loader=private_loader,
        privacy_engine=privacy_engine,
        noise_multiplier=noise_multiplier,
    )


def get_spent_epsilon(
    privacy_engine: PrivacyEngine,
    delta: float,
) -> float:
    """Return the privacy expenditure accumulated so far."""

    if not 0 < delta < 1:
        raise ValueError(
            "delta must be strictly between zero and one"
        )

    # No DP optimiser steps have occurred yet.
    if len(privacy_engine.accountant) == 0:
        return 0.0

    return float(
        privacy_engine.get_epsilon(delta=delta)
    )

def make_training_clipping_only(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    max_grad_norm: float,
) -> PrivateTrainingObjects:
    """
    Wrap training with Opacus per-sample gradient clipping
    but without Gaussian noise.

    This is an experimental ablation only and does NOT
    provide Differential Privacy.
    """

    if max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be greater than zero"
        )

    privacy_engine = PrivacyEngine(
        accountant="prv"
    )

    private_model, private_optimizer, private_loader = (
        privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
        )
    )

    return PrivateTrainingObjects(
        model=private_model,
        optimizer=private_optimizer,
        data_loader=private_loader,
        privacy_engine=privacy_engine,
        noise_multiplier=0.0,
    )

def make_training_clipping_poisson_no_noise(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    max_grad_norm: float,
) -> PrivateTrainingObjects:
    """
    Use Opacus per-sample clipping and Poisson sampling,
    but disable Gaussian noise.

    This is an ablation experiment only and does NOT
    provide a meaningful Differential Privacy guarantee.
    """

    if max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be greater than zero"
        )

    privacy_engine = PrivacyEngine(
        accountant="prv"
    )

    private_model, private_optimizer, private_loader = (
        privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
        )
    )

    return PrivateTrainingObjects(
        model=private_model,
        optimizer=private_optimizer,
        data_loader=private_loader,
        privacy_engine=privacy_engine,
        noise_multiplier=0.0,
    )