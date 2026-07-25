"""Standard episode summary and trajectory manager."""

from __future__ import annotations

import numpy as np

from dl_core.core import (
    BaseEpisodeManager,
    EpisodeRecord,
    EpisodeResult,
    register_episode_manager,
)


@register_episode_manager("standard")
class StandardEpisodeManager(BaseEpisodeManager):
    """Track generic returns, rewards, lengths, and selected trajectories."""

    def summarize_episode(
        self,
        record: EpisodeRecord,
        result: EpisodeResult,
        **statistics: float | int,
    ) -> dict[str, float]:
        """Return generic scalar metrics for one completed episode."""
        length = int(statistics["length"])
        episode_return = float(statistics["episode_return"])
        reward_square_sum = float(statistics["reward_square_sum"])
        reward_mean = episode_return / length if length else 0.0
        reward_variance = (
            max(0.0, reward_square_sum / length - reward_mean * reward_mean)
            if length
            else 0.0
        )
        metrics = {
            "episode/return": episode_return,
            "episode/length": float(length),
            "episode/reward_mean": reward_mean,
            "episode/reward_std": reward_variance**0.5,
            "episode/reward_min": (
                float(statistics["reward_min"]) if length else 0.0
            ),
            "episode/reward_max": (
                float(statistics["reward_max"]) if length else 0.0
            ),
            "episode/terminated": float(result.terminated),
            "episode/truncated": float(result.truncated),
        }
        success = result.final_info.get("is_success")
        if isinstance(success, (bool, int, float, np.bool_, np.number)):
            metrics["episode/success"] = float(success)
        return metrics
