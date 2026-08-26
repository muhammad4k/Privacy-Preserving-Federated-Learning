"""Sanity checks for Project Atlas CNN models."""

import torch

from models.cnn import create_model


def main() -> None:
    mnist_model = create_model("mnist")
    mnist_output = mnist_model(torch.randn(8, 1, 28, 28))

    fashion_model = create_model("fashion_mnist")
    fashion_output = fashion_model(torch.randn(8, 1, 28, 28))

    cifar_model = create_model("cifar10")
    cifar_output = cifar_model(torch.randn(8, 3, 32, 32))

    assert mnist_output.shape == (8, 10)
    assert fashion_output.shape == (8, 10)
    assert cifar_output.shape == (8, 10)

    print("MNIST output shape:", tuple(mnist_output.shape))
    print("Fashion-MNIST output shape:", tuple(fashion_output.shape))
    print("CIFAR-10 output shape:", tuple(cifar_output.shape))
    print("Model checks passed.")


if __name__ == "__main__":
    main()