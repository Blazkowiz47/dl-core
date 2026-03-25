"""Local tracker backend for file-based dl-core runs."""

from __future__ import annotations

from dl_core.core import BaseTracker, register_tracker


@register_tracker("local")
class LocalTracker(BaseTracker):
    """Default local tracker for file-based experiment metadata."""

    def get_backend_name(self) -> str:
        """Return the tracker backend name."""
        return "local"
