"""Base metrics source interface for sweep analysis backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml


class BaseMetricsSource(ABC):
    """
    Abstract base class for sweep analyzer metrics sources.

    Metrics sources translate backend-specific tracking data into the normalized
    run records consumed by ``dl-analyze-sweep``.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the metrics source.

        Args:
            **kwargs: Backend-specific parameters
        """
        self.params = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def collect_run(
        self,
        run_index: int,
        run_data: dict[str, Any],
        sweep_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a normalized analyzer record for one tracked run.

        Args:
            run_index: Sweep run index
            run_data: Per-run entry from ``sweep_tracking.json``
            sweep_data: Full sweep tracking payload

        Returns:
            Normalized run analysis record.
        """

    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        """
        Load a JSON file into a dictionary.

        Args:
            path: File to read

        Returns:
            Parsed JSON data.
        """
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_yaml(path: Path) -> dict[str, Any]:
        """
        Load a YAML file into a dictionary.

        Args:
            path: File to read

        Returns:
            Parsed YAML data, or an empty dictionary if the root is not a
            mapping.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
