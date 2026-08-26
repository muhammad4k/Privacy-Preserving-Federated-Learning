"""Result persistence utilities for Project Atlas."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from federated.simulator import SimulationResult


def save_simulation_result(
    result: SimulationResult,
    output_directory: str | Path = "results/raw",
    run_name: str | None = None,
) -> tuple[Path, Path]:
    """Save one simulation result as CSV and JSON."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{result.dataset_name}_"
            f"{result.num_clients}clients_"
            f"{result.num_rounds}rounds_"
            f"{timestamp}"
        )

    csv_path = output_path / f"{run_name}.csv"
    json_path = output_path / f"{run_name}.json"

    round_rows = [asdict(round_result) for round_result in result.rounds]
    dataframe = pd.DataFrame(round_rows)
    dataframe.to_csv(csv_path, index=False)

    json_payload = {
        "dataset_name": result.dataset_name,
        "num_clients": result.num_clients,
        "num_rounds": result.num_rounds,
        "rounds": round_rows,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=2)

    return csv_path, json_path