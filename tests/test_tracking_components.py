"""Tests for tracker and metrics source core abstractions."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import dl_core
from dl_core.analysis.sweep_analyzer import collect_sweep_runs
from dl_core.core import METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY
from dl_core.executors.local import LocalExecutor
from dl_core.sweep.template import generate_experiment_name


def test_local_tracker_is_registered_and_injects_tracking_config() -> None:
    """Local tracker should register and inject generic tracking metadata."""
    dl_core.load_builtin_components()

    assert TRACKER_REGISTRY.is_registered("local")

    tracker = TRACKER_REGISTRY.get("local", {"experiment_name": "demo-project"})
    config: dict[str, object] = {}
    tracker.inject_tracking_config(
        config,
        run_name="demo-run",
        tracking_context="demo-group",
        tracking_uri="./mlruns",
    )

    assert config["tracking"] == {
        "backend": "local",
        "context": "demo-group",
        "experiment_name": "demo-project",
        "uri": "./mlruns",
        "run_name": "demo-run",
    }


def test_local_tracker_builds_default_run_reference() -> None:
    """Local tracker should emit a minimal backend-specific run reference."""
    dl_core.load_builtin_components()

    tracker = TRACKER_REGISTRY.get("local")
    run_reference = tracker.build_run_reference(
        result={"tracking_run_name": "demo-run"},
        run_name="demo-run",
        tracking_context="demo-group",
        tracking_uri="./artifacts",
    )

    assert run_reference == {
        "backend": "local",
        "run_name": "demo-run",
        "tracking_context": "demo-group",
        "tracking_uri": "./artifacts",
    }


def test_collect_sweep_runs_uses_local_metrics_source(tmp_path: Path) -> None:
    """Sweep analyzer should resolve local metrics via the registered source."""
    dl_core.load_builtin_components()

    assert METRICS_SOURCE_REGISTRY.is_registered("local")

    experiments_dir = tmp_path / "experiments"
    sweep_path = experiments_dir / "lr_sweep.yaml"
    tracking_dir = experiments_dir / "lr_sweep"
    tracking_path = tracking_dir / "sweep_tracking.json"
    artifact_dir = tmp_path / "artifacts" / "demo-exp" / "lr_sweep" / "run_0"
    metrics_dir = artifact_dir / "metrics"
    summary_path = metrics_dir / "summary.json"
    history_path = metrics_dir / "history.json"
    config_path = tracking_dir / "run_0.yaml"

    tracking_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    sweep_path.write_text("grid:\n  optimizers.lr: [0.001]\n", encoding="utf-8")
    config_path.write_text(
        yaml.dump(
            {
                "experiment": {"name": "demo-exp"},
                "runtime": {"name": "demo-run", "output_dir": "artifacts"},
                "sweep_file": str(sweep_path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary_path.write_text(
        json.dumps(
            {
                "run_name": "demo-run",
                "selection_metric": "validation_accuracy",
                "selection_mode": "max",
                "selection_value": None,
                "best_epoch": 2,
                "final_epoch": 3,
                "best_metrics": {"validation/accuracy": 0.91},
                "final_metrics": {"validation/accuracy": 0.88},
            }
        ),
        encoding="utf-8",
    )
    history_path.write_text(json.dumps({"epochs": []}), encoding="utf-8")
    tracking_path.write_text(
        json.dumps(
            {
                "experiment_name": "demo-exp",
                "sweep_config": sweep_path.name,
                "sweep_id": "demo-sweep",
                "user": "tester",
                "total_runs": 1,
                "tracking_context": None,
                "tracking_backend": "local",
                "metrics_source_backend": "local",
                "runs": {
                    "0": {
                        "tracking_run_id": None,
                        "tracking_run_name": "demo-run",
                        "tracking_run_ref": {
                            "backend": "local",
                            "run_name": "demo-run",
                        },
                        "tracking_backend": "local",
                        "metrics_source_backend": "local",
                        "config_path": str(config_path),
                        "artifact_dir": str(artifact_dir),
                        "metrics_summary_path": str(summary_path),
                        "metrics_history_path": str(history_path),
                        "status": "completed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runs = collect_sweep_runs(sweep_path)

    assert len(runs) == 1
    assert runs[0]["run_name"] == "demo-run"
    assert runs[0]["status"] == "completed"
    assert runs[0]["selection_metric"] == "validation_accuracy"
    assert runs[0]["selection_value"] == 0.91
    assert runs[0]["metrics_summary_path"] == str(summary_path)


def test_generate_experiment_name_defaults_to_project_root(tmp_path: Path) -> None:
    """Tracker experiment naming should default to the repository root name."""
    project_root = tmp_path / "demo_repo"
    sweep_path = project_root / "experiments" / "lr_sweep.yaml"
    (project_root / "src").mkdir(parents=True)
    sweep_path.parent.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n")

    experiment_name = generate_experiment_name(
        {"sweep_file": str(sweep_path)},
        timestamp="",
    )

    assert experiment_name == "demo_repo"


def test_local_executor_injects_tracker_metadata_into_run_config() -> None:
    """Local executor should inject tracker metadata before launching a run."""
    dl_core.load_builtin_components()

    executor = LocalExecutor(
        {
            "tracking": {
                "backend": "local",
                "experiment_name": "demo-project",
            },
            "sweep_file": "experiments/demo_sweep.yaml",
        },
        experiment_name="demo-exp",
        sweep_id="sweep-001",
        tracking_context="demo-group",
    )
    executor.tracking_uri = "./artifacts"

    config: dict[str, object] = {"runtime": {"name": "demo-run"}}
    executor._inject_sweep_metadata(config)

    assert config["tracking"] == {
        "backend": "local",
        "context": "demo-group",
        "experiment_name": "demo-project",
        "uri": "./artifacts",
        "run_name": "demo-run",
    }
    assert config["sweep_file"] == "experiments/demo_sweep.yaml"
    assert config["auto_resume_local"] is True
