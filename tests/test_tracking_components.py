"""Tests for tracker and metrics source core abstractions."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pytest import CaptureFixture

import dl_core
from dl_core.analysis.sweep_analyzer import (
    collect_sweep_runs,
    get_sweep_analysis_markdown_path,
    main,
)
from dl_core.core import METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY
from dl_core.executors.local import LocalExecutor
from dl_core.sweep.template import generate_experiment_name
from dl_core.utils.artifact_manager import (
    get_run_artifact_dir,
    resolve_existing_run_artifact_dir,
)


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


def test_collect_sweep_runs_recovers_generated_config_path(tmp_path: Path) -> None:
    """Analyzer should recover the generated local run config when tracker data omits it."""
    dl_core.load_builtin_components()

    experiments_dir = tmp_path / "experiments"
    sweep_path = experiments_dir / "remote_sweep.yaml"
    tracking_dir = experiments_dir / "remote_sweep"
    tracking_path = tracking_dir / "sweep_tracking.json"
    config_path = tracking_dir / "azure_run.yaml"
    summary_path = tmp_path / "summary.json"
    history_path = tmp_path / "history.json"

    tracking_dir.mkdir(parents=True)
    sweep_path.write_text("grid:\n  optimizers.lr: [0.001]\n", encoding="utf-8")
    config_path.write_text(
        yaml.dump(
            {
                "experiment": {"name": "demo-exp"},
                "runtime": {"output_dir": "artifacts"},
                "callbacks": {
                    "checkpoint": {"monitor": "test/accuracy", "mode": "max"}
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "run_name": "azure_run",
                "selection_metric": "test/accuracy",
                "selection_mode": "max",
                "selection_value": 0.8,
                "best_epoch": 1,
                "final_metrics": {"test/accuracy": 0.8},
                "best_metrics": {"test/accuracy": 0.8},
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
                "sweep_id": "demo-remote",
                "user": "tester",
                "total_runs": 1,
                "tracking_backend": "local",
                "metrics_source_backend": "local",
                "runs": {
                    "0": {
                        "tracking_run_name": "azure_run",
                        "config_path": None,
                        "artifact_dir": None,
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

    assert runs[0]["config_path"] == str(config_path)


def test_dl_analyze_writes_markdown_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """Analyzer CLI should always write analysis.md next to the sweep outputs."""
    dl_core.load_builtin_components()

    experiments_dir = tmp_path / "experiments"
    sweep_path = experiments_dir / "lr_sweep.yaml"
    tracking_dir = experiments_dir / "lr_sweep"
    tracking_path = tracking_dir / "sweep_tracking.json"
    config_path = tracking_dir / "demo_run.yaml"
    summary_path = tmp_path / "summary.json"
    history_path = tmp_path / "history.json"

    tracking_dir.mkdir(parents=True)
    sweep_path.write_text("grid:\n  optimizers.lr: [0.001]\n", encoding="utf-8")
    config_path.write_text(
        yaml.dump(
            {
                "experiment": {"name": "demo-exp"},
                "runtime": {"output_dir": "artifacts"},
                "callbacks": {
                    "checkpoint": {"monitor": "validation/accuracy", "mode": "max"}
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "run_name": "demo_run",
                "selection_metric": "validation/accuracy",
                "selection_mode": "max",
                "selection_value": 0.91,
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
                "tracking_backend": "local",
                "metrics_source_backend": "local",
                "runs": {
                    "0": {
                        "tracking_run_name": "demo_run",
                        "config_path": str(config_path),
                        "artifact_dir": None,
                        "metrics_summary_path": str(summary_path),
                        "metrics_history_path": str(history_path),
                        "status": "completed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--sweep", str(sweep_path)])
    stdout = capsys.readouterr().out
    markdown_path = get_sweep_analysis_markdown_path(sweep_path)

    assert exit_code == 0
    assert "Wrote Markdown report to" in stdout
    assert markdown_path.exists()
    report = markdown_path.read_text(encoding="utf-8")
    assert "# Sweep Analysis" in report
    assert "demo_run" in report
    assert "validation/accuracy" in report


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


def test_generate_experiment_name_prefers_experiment_override(
    tmp_path: Path,
) -> None:
    """Explicit experiment.name should override the repository root default."""
    project_root = tmp_path / "demo_repo"
    sweep_path = project_root / "experiments" / "lr_sweep.yaml"
    (project_root / "src").mkdir(parents=True)
    sweep_path.parent.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n")

    experiment_name = generate_experiment_name(
        {
            "sweep_file": str(sweep_path),
            "experiment": {"name": "custom-exp"},
        },
        timestamp="",
    )

    assert experiment_name == "custom-exp"


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


def test_artifact_paths_use_flat_layout_with_legacy_fallback(
    tmp_path: Path,
) -> None:
    """Artifact helpers should prefer the new layout and still resolve old runs."""
    output_dir = tmp_path / "artifacts"
    new_run_dir = Path(
        get_run_artifact_dir(
            run_name="demo-run",
            output_dir=str(output_dir),
            experiment_name="demo-exp",
        )
    )
    assert new_run_dir == output_dir / "runs" / "demo-run"

    legacy_run_dir = output_dir / "demo-exp" / "demo-run"
    legacy_run_dir.mkdir(parents=True)

    resolved_legacy = resolve_existing_run_artifact_dir(
        run_name="demo-run",
        output_dir=str(output_dir),
        experiment_name="demo-exp",
    )
    assert resolved_legacy == legacy_run_dir

    new_run_dir.mkdir(parents=True)
    resolved_new = resolve_existing_run_artifact_dir(
        run_name="demo-run",
        output_dir=str(output_dir),
        experiment_name="demo-exp",
    )
    assert resolved_new == new_run_dir
