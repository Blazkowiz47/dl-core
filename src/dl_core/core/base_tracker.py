"""Base tracker interface for sweep and run metadata backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):
    """
    Abstract base class for tracking backends.

    Trackers are responsible for describing the external tracking backend used
    by a sweep and for injecting tracker-specific metadata into generated run
    configurations. They intentionally do not own execution.
    """

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
            tracking_context: Optional tracker-specific group or parent context
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
