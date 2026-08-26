"""Build consolidated dissertation results for Project Atlas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")
FIGURES_DIR = Path("figures")

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Experiments used in the main dissertation comparison
# ---------------------------------------------------------------------

EXPERIMENTS = [
    {
        "file": "baseline_mnist_15rounds.csv",
        "experiment": "MNIST Baseline",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "Non-private",
        "target_epsilon": None,
        "clip_norm": None,
    },
    {
        "file": "final_dp_mnist_c1_0_eps10.csv",
        "experiment": "MNIST DP C1 eps10",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 10.0,
        "clip_norm": 1.0,
    },
    {
        "file": "final_dp_mnist_c1_0_eps3.csv",
        "experiment": "MNIST DP C1 eps3",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 3.0,
        "clip_norm": 1.0,
    },
    {
        "file": "final_dp_mnist_c1_0_eps1.csv",
        "experiment": "MNIST DP C1 eps1",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 1.0,
        "clip_norm": 1.0,
    },
    {
        "file": "final_dp_mnist_c5_0_eps10.csv",
        "experiment": "MNIST DP C5 eps10",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 10.0,
        "clip_norm": 5.0,
    },
    {
        "file": "final_dp_mnist_c5_0_eps3.csv",
        "experiment": "MNIST DP C5 eps3",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 3.0,
        "clip_norm": 5.0,
    },
    {
        "file": "final_dp_mnist_c5_0_eps1.csv",
        "experiment": "MNIST DP C5 eps1",
        "dataset": "MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 1.0,
        "clip_norm": 5.0,
    },
    {
        "file": "baseline_mnist_non_iid_15rounds.csv",
        "experiment": "MNIST Non-IID Baseline",
        "dataset": "MNIST",
        "partition": "Non-IID",
        "privacy": "Non-private",
        "target_epsilon": None,
        "clip_norm": None,
    },
    {
        "file": "dp_mnist_non_iid_c5_eps3.csv",
        "experiment": "MNIST Non-IID DP eps3",
        "dataset": "MNIST",
        "partition": "Non-IID",
        "privacy": "DP",
        "target_epsilon": 3.0,
        "clip_norm": 5.0,
    },
    {
        "file": "dp_mnist_non_iid_c5_eps1.csv",
        "experiment": "MNIST Non-IID DP eps1",
        "dataset": "MNIST",
        "partition": "Non-IID",
        "privacy": "DP",
        "target_epsilon": 1.0,
        "clip_norm": 5.0,
    },
    {
        "file": "baseline_fashion_mnist_15rounds.csv",
        "experiment": "Fashion-MNIST Baseline",
        "dataset": "Fashion-MNIST",
        "partition": "IID",
        "privacy": "Non-private",
        "target_epsilon": None,
        "clip_norm": None,
    },
    {
        "file": "dp_fashion_mnist_c5_eps3.csv",
        "experiment": "Fashion-MNIST DP eps3",
        "dataset": "Fashion-MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 3.0,
        "clip_norm": 5.0,
    },
    {
        "file": "dp_fashion_mnist_c5_eps1.csv",
        "experiment": "Fashion-MNIST DP eps1",
        "dataset": "Fashion-MNIST",
        "partition": "IID",
        "privacy": "DP",
        "target_epsilon": 1.0,
        "clip_norm": 5.0,
    },
]


def get_column(frame: pd.DataFrame, candidates: list[str]):
    """Return the first matching column name."""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def extract_final_result(metadata: dict) -> dict:
    """Extract the final-round metrics from one experiment."""

    path = RESULTS_DIR / metadata["file"]

    if not path.exists():
        print(f"WARNING: missing {path}")
        return {}

    frame = pd.read_csv(path)

    if frame.empty:
        print(f"WARNING: empty {path}")
        return {}

    final = frame.iloc[-1]

    accuracy_col = get_column(
        frame,
        ["accuracy", "global_accuracy"],
    )

    f1_col = get_column(
        frame,
        ["macro_f1", "global_macro_f1"],
    )

    loss_col = get_column(
        frame,
        ["test_loss", "global_loss", "loss"],
    )

    epsilon_col = get_column(
        frame,
        [
            "mean_client_epsilon",
            "max_client_epsilon",
            "mean_epsilon",
            "epsilon",
            "max_epsilon",
        ],
    )

    noise_col = get_column(
        frame,
        ["noise_multiplier", "mean_noise_multiplier"],
    )

    return {
        **metadata,
        "final_accuracy": (
            float(final[accuracy_col])
            if accuracy_col is not None
            else None
        ),
        "final_macro_f1": (
            float(final[f1_col])
            if f1_col is not None
            else None
        ),
        "final_test_loss": (
            float(final[loss_col])
            if loss_col is not None
            else None
        ),
        "actual_epsilon": (
            float(final[epsilon_col])
            if epsilon_col is not None
            and pd.notna(final[epsilon_col])
            else None
        ),
        "noise_multiplier": (
            float(final[noise_col])
            if noise_col is not None
            and pd.notna(final[noise_col])
            else None
        ),
    }


rows = []

for experiment in EXPERIMENTS:
    result = extract_final_result(experiment)

    if result:
        rows.append(result)


master = pd.DataFrame(rows)

if master.empty:
    raise RuntimeError(
        "No experiment results were found. "
        "Check the filenames in results/raw."
    )


# ---------------------------------------------------------------------
# Save master table
# ---------------------------------------------------------------------

master_path = SUMMARY_DIR / "master_results.csv"
master.to_csv(master_path, index=False)

print("\n" + "=" * 72)
print("PROJECT ATLAS MASTER RESULTS")
print("=" * 72)

display_columns = [
    "experiment",
    "dataset",
    "partition",
    "privacy",
    "target_epsilon",
    "clip_norm",
    "actual_epsilon",
    "noise_multiplier",
    "final_accuracy",
    "final_macro_f1",
    "final_test_loss",
]

print(master[display_columns].to_string(index=False))
print(f"\nSaved: {master_path}")


# ---------------------------------------------------------------------
# Figure 1: MNIST privacy-utility comparison
# ---------------------------------------------------------------------

mnist_dp = master[
    (master["dataset"] == "MNIST")
    & (master["partition"] == "IID")
    & (master["privacy"] == "DP")
].copy()

if not mnist_dp.empty:

    plt.figure(figsize=(7, 5))

    for clip_norm in sorted(
        mnist_dp["clip_norm"].dropna().unique()
    ):
        subset = mnist_dp[
            mnist_dp["clip_norm"] == clip_norm
        ].sort_values("actual_epsilon")

        plt.plot(
            subset["actual_epsilon"],
            subset["final_accuracy"],
            marker="o",
            label=f"C={clip_norm:g}",
        )

    plt.xlabel("Privacy budget (epsilon)")
    plt.ylabel("Final test accuracy")
    plt.title("MNIST Privacy-Utility Trade-off")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    path = FIGURES_DIR / "master_privacy_utility.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 2: IID vs Non-IID
# ---------------------------------------------------------------------

iid_non_iid = master[
    (master["dataset"] == "MNIST")
    & (master["clip_norm"].fillna(5.0) == 5.0)
].copy()

labels = []
accuracies = []

for partition in ["IID", "Non-IID"]:

    baseline = iid_non_iid[
        (iid_non_iid["partition"] == partition)
        & (iid_non_iid["privacy"] == "Non-private")
    ]

    if not baseline.empty:
        labels.append(f"{partition}\nBaseline")
        accuracies.append(
            baseline.iloc[0]["final_accuracy"]
        )

    for epsilon in [3.0, 1.0]:
        row = iid_non_iid[
            (iid_non_iid["partition"] == partition)
            & (iid_non_iid["privacy"] == "DP")
            & (iid_non_iid["target_epsilon"] == epsilon)
        ]

        if not row.empty:
            labels.append(f"{partition}\nε={epsilon:g}")
            accuracies.append(
                row.iloc[0]["final_accuracy"]
            )

if accuracies:

    plt.figure(figsize=(8, 5))
    plt.bar(labels, accuracies)

    plt.ylabel("Final test accuracy")
    plt.title("MNIST IID vs Non-IID Performance")
    plt.ylim(0, 1)
    plt.tight_layout()

    path = FIGURES_DIR / "iid_vs_non_iid.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 3: Clipping-only sensitivity
# ---------------------------------------------------------------------

clipping_files = {
    0.5: RESULTS_DIR / "clipping_only_mnist_c0_5.csv",
    1.0: RESULTS_DIR / "clipping_only_mnist_c1_0.csv",
    2.0: RESULTS_DIR / "clipping_only_mnist_c2_0.csv",
    5.0: RESULTS_DIR / "clipping_only_mnist_c5_0.csv",
}

clip_norms = []
clip_accuracies = []

for clip_norm, file_path in clipping_files.items():
    if file_path.exists():
        df = pd.read_csv(file_path)
        final_row = df.iloc[-1]

        clip_norms.append(clip_norm)
        clip_accuracies.append(
            float(final_row["global_accuracy"])
        )

if clip_norms:

    plt.figure(figsize=(7, 5))

    plt.plot(
        clip_norms,
        clip_accuracies,
        marker="o",
    )

    plt.xlabel("Maximum gradient norm (C)")
    plt.ylabel("Final test accuracy")
    plt.title("Effect of Gradient Clipping on MNIST Utility")
    plt.xticks(clip_norms)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    path = FIGURES_DIR / "clipping_sensitivity_summary.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 4: Cross-dataset comparison
# ---------------------------------------------------------------------

cross_dataset = master[
    (
        (master["dataset"] == "MNIST")
        | (master["dataset"] == "Fashion-MNIST")
    )
    & (master["partition"] == "IID")
].copy()

labels = []
accuracies = []

for dataset in ["MNIST", "Fashion-MNIST"]:

    baseline = cross_dataset[
        (cross_dataset["dataset"] == dataset)
        & (cross_dataset["privacy"] == "Non-private")
    ]

    if not baseline.empty:
        labels.append(f"{dataset}\nBaseline")
        accuracies.append(
            baseline.iloc[0]["final_accuracy"]
        )

    for epsilon in [3.0, 1.0]:
        row = cross_dataset[
            (cross_dataset["dataset"] == dataset)
            & (cross_dataset["privacy"] == "DP")
            & (
                cross_dataset["target_epsilon"]
                == epsilon
            )
            & (
                cross_dataset["clip_norm"]
                == 5.0
            )
        ]

        if not row.empty:
            labels.append(f"{dataset}\nε={epsilon:g}")
            accuracies.append(
                row.iloc[0]["final_accuracy"]
            )

if accuracies:

    plt.figure(figsize=(9, 5))
    plt.bar(labels, accuracies)

    plt.ylabel("Final test accuracy")
    plt.title(
        "Cross-Dataset Privacy-Utility Comparison"
    )
    plt.ylim(0, 1)
    plt.tight_layout()

    path = FIGURES_DIR / "cross_dataset_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


print("\nMaster results build completed.")


