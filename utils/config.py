"""Configuration loader for Project Atlas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Loads the YAML configuration file and provides
    easy access to experiment settings.
    """

    def __init__(self, config_path: str | Path):

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)

    def get(self, *keys: str) -> Any:
        """
        Retrieve nested configuration values.

        Example:

        config.get("training", "batch_size")
        """

        value = self._config

        for key in keys:

            if key not in value:
                raise KeyError(
                    f"Configuration key not found: {' -> '.join(keys)}"
                )

            value = value[key]

        return value

    @property
    def raw(self):

        return self._config