"""Tests for the local JSONL metric tracker callback."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dl_core.callbacks.local_metric_tracker import LocalMetricTrackerCallback
from dl_core.utils.artifact_manager import ArtifactManager


class _DummyAccelerator:
    """Simple accelerator test double."""

    def is_main_process(self) -> bool:
        """Report that this process is the main process."""
        return True


class _DummyTrainer:
    """Small trainer test double with an artifact manager."""

    def __init__(self, artifact_manager: ArtifactManager) -> None:
        self.accelerator = _DummyAccelerator()
        self.artifact_manager = artifact_manager


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file into a list of dictionaries."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_local_metric_tracker_callback_appends_per_metric_jsonl() -> None:
    """The callback should append phase and general metrics with correct steps."""
    with TemporaryDirectory() as temp_dir:
        artifact_manager = ArtifactManager(
            run_name="demo-run",
            output_dir=temp_dir,
            experiment_name="demo-exp",
        )
        callback = LocalMetricTrackerCallback()
        callback.set_trainer(_DummyTrainer(artifact_manager))

        callback.on_test_end(
            0,
            {
                "accuracy": 0.61,
                "note": "ignored",
            },
        )
        callback.on_train_end(
            1,
            {
                "loss": 0.5,
            },
        )
        callback.on_validation_end(
            1,
            {
                "accuracy": 0.75,
            },
        )
        callback.on_epoch_end(
            1,
            {
                "general/state/global_step": 32.0,
                "note": "ignored",
            },
        )
        callback.on_episode_end(
            2,
            {"episode/return": 4.5, "global_step": 21},
        )
        callback.on_update_end(
            3,
            {"dqn/loss": 0.2, "global_step": 22},
        )
        callback.on_evaluation_end(
            22,
            {"evaluation/mean_return": 5.0, "global_step": 22},
        )

        series_dir = artifact_manager.get_metric_streams_dir()
        test_accuracy_records = _read_jsonl(series_dir / "test_accuracy.jsonl")
        train_loss_records = _read_jsonl(series_dir / "train_loss.jsonl")
        validation_records = _read_jsonl(series_dir / "validation_accuracy.jsonl")
        global_step_records = _read_jsonl(
            series_dir / "general_state_global_step.jsonl"
        )
        episode_records = _read_jsonl(series_dir / "episode_return.jsonl")
        update_records = _read_jsonl(series_dir / "dqn_loss.jsonl")
        evaluation_records = _read_jsonl(
            series_dir / "evaluation_mean_return.jsonl"
        )

        assert test_accuracy_records == [
            {"metric": "test/accuracy", "step": 0, "epoch": 0, "value": 0.61}
        ]
        assert train_loss_records == [
            {"metric": "train/loss", "step": 1, "epoch": 1, "value": 0.5}
        ]
        assert validation_records == [
            {
                "metric": "validation/accuracy",
                "step": 1,
                "epoch": 1,
                "value": 0.75,
            },
        ]
        assert global_step_records == [
            {
                "metric": "general/state/global_step",
                "step": 1,
                "epoch": 1,
                "value": 32.0,
            }
        ]
        assert episode_records == [
            {
                "metric": "episode/return",
                "step": 21,
                "episode": 2,
                "value": 4.5,
            }
        ]
        assert update_records == [
            {
                "metric": "dqn/loss",
                "step": 22,
                "update": 3,
                "value": 0.2,
            }
        ]
        assert evaluation_records == [
            {
                "metric": "evaluation/mean_return",
                "step": 22,
                "global_step": 22,
                "value": 5.0,
            }
        ]
