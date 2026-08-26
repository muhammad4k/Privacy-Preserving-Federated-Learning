"""Integration test for Project Atlas training."""

from torch.utils.data import DataLoader

from datasets.loaders import load_dataset
from models.cnn import create_model
from training.trainer import (
    create_optimizer,
    evaluate_model,
    get_device,
    train_model,
)


def main():

    print("=" * 60)
    print("PROJECT ATLAS - TRAINING TEST")
    print("=" * 60)

    train_dataset, test_dataset = load_dataset("mnist")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    device = get_device()

    print(f"Device: {device}")

    model = create_model("mnist")

    optimizer = create_optimizer(
        model=model,
        learning_rate=0.01,
        momentum=0.9,
    )

    print("\nEvaluating BEFORE training...")

    before = evaluate_model(
        model,
        test_loader,
        device,
    )

    print(f"Accuracy: {before.accuracy:.4f}")

    print("\nTraining for ONE local epoch...")

    result = train_model(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        local_epochs=1,
        device=device,
    )

    print(f"Training Loss: {result.average_loss:.4f}")

    print("\nEvaluating AFTER training...")

    after = evaluate_model(
        model,
        test_loader,
        device,
    )

    print(f"Accuracy: {after.accuracy:.4f}")

    print("\nMacro F1:", round(after.macro_f1, 4))

    print("=" * 60)

    assert after.accuracy > before.accuracy

    print("TRAINING TEST PASSED")


if __name__ == "__main__":
    main()