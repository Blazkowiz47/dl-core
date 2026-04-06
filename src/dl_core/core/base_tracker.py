"""Base tracker interface for sweep and run metadata backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from dl_core.core.config_metadata import config_field


class BaseTracker(ABC):
    """
    Abstract base class for tracking backends.

    Trackers are responsible for describing the external tracking backend used
    by a sweep and for injecting tracker-specific metadata into generated run
    configurations. They intentionally do not own execution.
    """

    CONFIG_FIELDS = [
        config_field(
            "experiment_name",
            "str | None",
            "Optional tracker-level experiment or project override for the sweep.",
            default=None,
        )
    ]

    def __init__(self, tracking_config: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the tracker.

        Args:
            tracking_config: Optional tracking configuration block
            **kwargs: Backend-specific overrides
        """
        self.tracking_config = dict(tracking_config or {})
        self.params = {**self.tracking_config, **kwargs}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_backend_name(self) -> str:
        """
        Return the canonical tracker backend name.

        Returns:
            Registry-facing tracker backend name.
        """

    def get_metrics_source_name(self) -> str:
        """
        Return the default metrics source backend for this tracker.

        Returns:
            Metrics source backend name.
        """
        return self.get_backend_name()

    def setup_sweep(
        self,
        *,
        experiment_name: str,
        sweep_id: str,
        sweep_config: dict[str, Any],
        total_runs: int,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """
        Prepare tracker state for a sweep.

        Args:
            experiment_name: Sweep experiment name
            sweep_id: Sweep identifier
            sweep_config: Full sweep configuration
            total_runs: Number of concrete runs in the sweep
            tracking_context: Existing tracker-specific parent or sweep context
            tracking_uri: Existing tracker endpoint or workspace URI
            resume: Whether the sweep is resuming an existing context

        Returns:
            Updated tracker state. Supported keys are ``tracking_context`` and
            ``tracking_uri``.
        """
        del experiment_name
        del sweep_id
        del sweep_config
        del total_runs
        del resume
        return {
            "tracking_context": tracking_context,
            "tracking_uri": tracking_uri,
        }

    def teardown_sweep(self) -> None:
        """Clean up any tracker-owned sweep state."""

    def build_run_reference(
        self,
        *,
        result: dict[str, Any] | None = None,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Build one backend-specific run reference for sweep tracking.

        Args:
            result: Executor result payload
            run_name: Generated run name
            tracking_context: Tracker-specific parent or sweep context
            tracking_uri: Tracker endpoint or workspace URI

        Returns:
            A backend-specific run reference dictionary, or ``None`` when the
            tracker has no external reference to persist.
        """
        result = result or {}
        reference = result.get("tracking_run_ref")
        if isinstance(reference, dict):
            merged_reference = dict(reference)
        else:
            merged_reference = {}

        if not merged_reference:
            merged_reference["backend"] = self.get_backend_name()

        tracking_run_id = result.get("tracking_run_id")
        if isinstance(tracking_run_id, str) and tracking_run_id:
            merged_reference.setdefault("run_id", tracking_run_id)

        resolved_run_name = result.get("tracking_run_name") or run_name
        if isinstance(resolved_run_name, str) and resolved_run_name:
            merged_reference.setdefault("run_name", resolved_run_name)

        if tracking_context:
            merged_reference.setdefault("tracking_context", tracking_context)

        if tracking_uri:
            merged_reference.setdefault("tracking_uri", tracking_uri)

        return merged_reference or None

    def inject_tracking_config(
        self,
        config: dict[str, Any],
        *,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """
        Inject tracker metadata into a run configuration.

        Args:
            config: Run configuration to modify in place
            run_name: Optional generated run name
            tracking_context: Optional tracker-specific sweep or parent context
            tracking_uri: Optional tracker endpoint or workspace URI
        """
        tracking = config.setdefault("tracking", {})
        tracking.setdefault("backend", self.get_backend_name())

        if tracking_context:
            tracking["context"] = tracking_context
        if tracking_uri:
            tracking["uri"] = tracking_uri
        if run_name:
            tracking["run_name"] = run_name

        experiment_name = self.tracking_config.get("experiment_name")
        if isinstance(experiment_name, str) and experiment_name:
            tracking.setdefault("experiment_name", experiment_name)
