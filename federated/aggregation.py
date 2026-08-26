"""Federated aggregation utilities for Project Atlas."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn


StateDictionary = OrderedDict[str, torch.Tensor]


def get_model_parameters(model: nn.Module) -> StateDictionary:
    """Return a detached CPU copy of a model state dictionary."""

    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )


def set_model_parameters(
    model: nn.Module,
    parameters: StateDictionary,
) -> None:
    """Load model parameters into a PyTorch module."""

    model.load_state_dict(parameters, strict=True)


def weighted_fedavg(
    client_parameters: Sequence[StateDictionary],
    client_num_examples: Sequence[int],
) -> StateDictionary:
    """
    Aggregate client models using sample-weighted Federated Averaging.

    Clients containing more training examples contribute proportionally
    more to the resulting global model.
    """

    if not client_parameters:
        raise ValueError("At least one client model is required")

    if len(client_parameters) != len(client_num_examples):
        raise ValueError(
            "client_parameters and client_num_examples must have equal length"
        )

    if any(num_examples <= 0 for num_examples in client_num_examples):
        raise ValueError("Every client must contain at least one example")

    reference_keys = list(client_parameters[0].keys())

    for parameters in client_parameters[1:]:
        if list(parameters.keys()) != reference_keys:
            raise ValueError("All client state dictionaries must match")

    total_examples = sum(client_num_examples)
    aggregated = OrderedDict()

    for parameter_name in reference_keys:
        reference_tensor = client_parameters[0][parameter_name]

        if not reference_tensor.is_floating_point():
            aggregated[parameter_name] = reference_tensor.clone()
            continue

        weighted_sum = torch.zeros_like(
            reference_tensor,
            dtype=torch.float32,
        )

        for parameters, num_examples in zip(
            client_parameters,
            client_num_examples,
        ):
            weight = num_examples / total_examples
            weighted_sum += parameters[parameter_name].float() * weight

        aggregated[parameter_name] = weighted_sum.to(reference_tensor.dtype)

    return aggregated