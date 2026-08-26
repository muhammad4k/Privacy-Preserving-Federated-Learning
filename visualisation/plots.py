"""Publication-quality plotting utilities for Project Atlas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_simulation_metrics(
    csv_path: str | Path,
    output_directory: str | Path = "figures",
    run_name: str | None = None,
) -> tuple[Path, Path, Path]:
    """Generate accuracy, loss, and macro-F1 plots from a result CSV."""

    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_file}")

    dataframe = pd.read_csv(csv_file)

    required_columns = {
        "round_number",
        "global_accuracy",
        "global_loss",
        "global_macro_f1",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}"
        )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = csv_file.stem

    accuracy_path = output_path / f"{run_name}_accuracy.png"
    loss_path = output_path / f"{run_name}_loss.png"
    f1_path = output_path / f"{run_name}_macro_f1.png"

    rounds = dataframe["round_number"]

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        dataframe["global_accuracy"] * 100,
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.title("Global Accuracy Across Federated Rounds")
    plt.xticks(rounds)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        dataframe["global_loss"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Global Test Loss")
    plt.title("Global Test Loss Across Federated Rounds")
    plt.xticks(rounds)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        dataframe["global_macro_f1"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Macro-F1")
    plt.title("Global Macro-F1 Across Federated Rounds")
    plt.xticks(rounds)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close()

    return accuracy_path, loss_path, f1_path