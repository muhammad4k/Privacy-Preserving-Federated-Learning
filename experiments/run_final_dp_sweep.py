"""Run the final MNIST IID Differential Privacy experiment matrix."""

from __future__ import annotations

from pathlib import Path
import csv
import time

from experiments.run_experiment import run_experiment


CONFIG_PATHS = [
    "configs/final_dp_mnist_c1_0_eps10.yaml",
    "configs/final_dp_mnist_c1_0_eps3.yaml",
    "configs/final_dp_mnist_c1_0_eps1.yaml",
    "configs/final_dp_mnist_c5_0_eps10.yaml",
    "configs/final_dp_mnist_c5_0_eps3.yaml",
    "configs/final_dp_mnist_c5_0_eps1.yaml",
]


def main() -> None:
    """Run all final DP experiments sequentially."""

    summary_rows = []

    print("=" * 80)
    print("FINAL MNIST IID DIFFERENTIAL PRIVACY SWEEP")
    print("=" * 80)

    for index, config_path in enumerate(
        CONFIG_PATHS,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(
            f"Experiment {index}/{len(CONFIG_PATHS)}"
        )
        print(config_path)
        print("=" * 80)

        start = time.perf_counter()

        result = run_experiment(
            config_path
        )

        runtime = (
            time.perf_counter()
            - start
        )

        final_round = result.rounds[-1]

        target_epsilon = result.target_epsilon
        clipping_norm = result.max_grad_norm

        achieved_epsilon = (
            final_round.max_client_epsilon
        )

        summary_rows.append(
            {
                "config": config_path,
                "clipping_norm":
                    clipping_norm,
                "target_epsilon":
                    target_epsilon,
                "achieved_epsilon":
                    achieved_epsilon,
                "final_accuracy":
                    final_round.global_accuracy,
                "final_macro_f1":
                    final_round.global_macro_f1,
                "final_test_loss":
                    final_round.global_loss,
                "mean_client_runtime_seconds":
                    final_round.mean_client_runtime_seconds,
                "total_experiment_runtime_seconds":
                    runtime,
                "mean_noise_multiplier":
                    final_round.mean_noise_multiplier,
            }
        )

        print("\nFINAL EXPERIMENT RESULT")
        print(
            f"C={clipping_norm}, "
            f"target epsilon={target_epsilon}"
        )
        print(
            f"Achieved epsilon: "
            f"{achieved_epsilon:.4f}"
        )
        print(
            f"Accuracy: "
            f"{final_round.global_accuracy:.4f}"
        )
        print(
            f"Macro-F1: "
            f"{final_round.global_macro_f1:.4f}"
        )
        print(
            f"Test loss: "
            f"{final_round.global_loss:.4f}"
        )
        print(
            f"Noise multiplier: "
            f"{final_round.mean_noise_multiplier:.4f}"
        )

    output_path = Path(
        "results/raw/final_dp_mnist_sweep_summary.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print("\n" + "=" * 80)
    print("FINAL DP SWEEP SUMMARY")
    print("=" * 80)

    for row in summary_rows:
        print(
            f"C={row['clipping_norm']:<4} | "
            f"target eps="
            f"{row['target_epsilon']:<4} | "
            f"actual eps="
            f"{row['achieved_epsilon']:.4f} | "
            f"noise="
            f"{row['mean_noise_multiplier']:.4f} | "
            f"accuracy="
            f"{row['final_accuracy']:.4f} | "
            f"macro_f1="
            f"{row['final_macro_f1']:.4f}"
        )

    print(
        "\nSaved summary:",
        output_path,
    )


if __name__ == "__main__":
    main()