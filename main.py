"""Command-line entry point for the privacy-preserving FL framework."""

from __future__ import annotations

import argparse

from experiments.run_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a federated-learning experiment from a YAML configuration."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/default.yaml",
        help="Path to an experiment YAML configuration.",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
