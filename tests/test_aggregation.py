"""Sanity checks for sample-weighted Federated Averaging."""

import torch

from federated.aggregation import weighted_fedavg
from models.cnn import create_model


def main() -> None:
    model_a = create_model("mnist")
    model_b = create_model("mnist")

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    for tensor in state_a.values():
        if tensor.is_floating_point():
            tensor.fill_(1.0)

    for tensor in state_b.values():
        if tensor.is_floating_point():
            tensor.fill_(3.0)

    aggregated = weighted_fedavg(
        client_parameters=[state_a, state_b],
        client_num_examples=[1, 3],
    )

    for tensor in aggregated.values():
        if tensor.is_floating_point():
            expected = torch.full_like(tensor, 2.5)
            assert torch.allclose(tensor, expected)

    print("Expected weighted value: 2.5")
    print("Federated aggregation checks passed.")


if __name__ == "__main__":
    main()