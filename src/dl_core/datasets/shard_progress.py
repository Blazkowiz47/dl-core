"""Training-process consumption counters for explicitly planned shard passes."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


class ShardProgress:
    """Count completed-batch samples, with optional eligible counts or budgets."""

    def __init__(self, totals: Mapping[str, int | None]) -> None:
        for total in totals.values():
            if total is not None and (
                isinstance(total, bool) or not isinstance(total, int) or total < 1
            ):
                raise ValueError("Shard totals must be positive integers or None")
        self._totals = dict(totals)
        self._consumed = dict.fromkeys(totals, 0)
        self._owner_pid = os.getpid()

    def record(self, shard_ids: Iterable[str]) -> None:
        """Record the shard identities from one completed training batch."""
        if os.getpid() != self._owner_pid:
            raise RuntimeError("Record shard progress in the owning training process")
        if isinstance(shard_ids, str):
            raise TypeError("Pass a collection of shard IDs, not a string")
        counts = Counter(shard_ids)
        missing = counts.keys() - self._totals.keys()
        if missing:
            raise KeyError(f"Unplanned shard IDs: {sorted(missing)}")
        for shard, count in counts.items():
            self._consumed[shard] += count

    def get(self, shard_id: str | None = None) -> dict[str, Any]:
        """Return counts and a fraction; an unknown total has fraction None."""
        if os.getpid() != self._owner_pid:
            raise RuntimeError("Read shard progress in the owning training process")
        result = {}
        for shard in self._totals if shard_id is None else [shard_id]:
            total = self._totals[shard]
            consumed = self._consumed[shard]
            result[shard] = {
                "consumed": consumed,
                "total": total,
                "fraction": min(1.0, consumed / total) if total else None,
            }
        return result if shard_id is None else result[shard_id]
