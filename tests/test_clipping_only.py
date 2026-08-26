"""Ablation test for Opacus per-sample clipping without noise."""

from datasets.loaders import load_dataset
from datasets.partitioning import partition_iid
from models.cnn import create_model
from privacy.dp_engine import make_training_clipping_only
from training.trainer import (
    create_optimizer,
    evaluate_model,
    get_device,
    train_model,
)
from torch.utils.data import DataLoader
from utils.reproducibility import set_global_seed


def main() -> None:
    set_global_seed(42)

    train_dataset, test_dataset = load_dataset("mnist")

    client_subsets = partition_iid(
        dataset=train_dataset,
        num_clients=5,
        seed=42,
    )

    train_loader = DataLoader(
        client_subsets[0],
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    model = create_model("mnist")

    optimizer = create_optimizer(
        model=model,
        learning_rate=0.01,
        momentum=0.9,
    )

    private_objects = make_training_clipping_only(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=1.0,
    )

    before = evaluate_model(
        private_objects.model,
        test_loader,
        get_device(),
    )

    result = train_model(
        model=private_objects.model,
        data_loader=private_objects.data_loader,
        optimizer=private_objects.optimizer,
        local_epochs=1,
        device=get_device(),
    )

    after = evaluate_model(
        private_objects.model,
        test_loader,
        get_device(),
    )

    print(
        "Accuracy before clipping-only training:",
        round(before.accuracy, 4),
    )

    print(
        "Training loss:",
        round(result.average_loss, 4),
    )

    print(
        "Accuracy after clipping-only training:",
        round(after.accuracy, 4),
    )

    print(
        "Noise multiplier:",
        private_objects.noise_multiplier,
    )

    assert private_objects.noise_multiplier == 0.0
    assert after.accuracy > before.accuracy

    print(
        "Clipping-only ablation checks passed."
    )


if __name__ == "__main__":
    main()