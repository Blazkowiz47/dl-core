"""End-to-end smoke tests for scaffolded experiment repositories."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml


def _entrypoint_path(name: str) -> Path:
    """Resolve a console entrypoint installed for the active interpreter."""
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        entrypoint = Path(virtual_env) / "bin" / name
        if entrypoint.exists():
            return entrypoint

    resolved = shutil.which(name)
    if resolved:
        return Path(resolved)

    entrypoint = Path(sys.executable).resolve().parent / name
    if entrypoint.exists():
        return entrypoint

    raise FileNotFoundError(f"Console entrypoint not found: {name}")


def _run_entrypoint(
    entrypoint: str,
    *args: str,
    cwd: Path,
    timeout_seconds: int = 180,
) -> str:
    """Run a dl-core console entrypoint and fail with captured output."""
    result = subprocess.run(
        [str(_entrypoint_path(entrypoint)), *args],
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{entrypoint} failed with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def _write_dummy_dataset_wrapper(repo_dir: Path, dataset_name: str) -> None:
    """Replace the scaffolded dataset wrapper with a local dummy dataset."""
    dataset_path = repo_dir / "src" / "datasets" / f"{dataset_name}.py"
    class_name = "".join(part.capitalize() for part in dataset_name.split("_"))
    dataset_path.write_text(
        f'''"""Local dummy dataset used for scaffold smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dl_core.core import register_dataset
from dl_core.datasets.standard import StandardWrapper


@register_dataset("{dataset_name}")
class {class_name}Dataset(StandardWrapper):
    """Local dummy dataset wrapper for end-to-end smoke tests."""

    _SPLIT_SIZES = {{
        "train": 12,
        "validation": 4,
        "test": 4,
    }}

    _SPLIT_OFFSETS = {{
        "train": 0,
        "validation": 1000,
        "test": 2000,
    }}

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return deterministic in-memory sample metadata for the requested split."""
        num_samples = self._SPLIT_SIZES.get(split, 0)
        classes = self.classes or [str(idx) for idx in range(self.num_classes or 2)]
        num_classes = len(classes)
        if num_classes == 0:
            raise ValueError("Dummy dataset requires at least one class label")

        return [
            {{
                "index": idx,
                "label": classes[idx % num_classes],
                "path": f"dummy://{{split}}/{{idx}}",
            }}
            for idx in range(num_samples)
        ]

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Create a deterministic dummy image tensor for the requested sample."""
        label = file_dict["label"]
        class_label = self.classes.index(label)
        offset = self._SPLIT_OFFSETS.get(split, 0)
        rng = np.random.default_rng(self.seed + offset + file_dict["index"])
        image = rng.integers(
            0,
            256,
            size=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        if self.augmentation:
            image_tensor = self.augmentation.apply(image, split)
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {{
            "image": image_tensor.float(),
            "label": class_label,
            "class": label,
            "path": str(file_dict["path"]),
        }}
''',
        encoding="utf-8",
    )


def _update_base_config(repo_dir: Path, dataset_name: str) -> None:
    """Shrink the scaffolded config so the smoke run stays fast on CPU."""
    config_path = repo_dir / "configs" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config["runtime"]["name"] = "scaffold_smoke"
    config["runtime"]["log_level"] = "WARNING"
    config["dataset"]["name"] = dataset_name
    config["dataset"]["height"] = 32
    config["dataset"]["width"] = 32
    config["dataset"]["batch_size"] = 4
    config["dataset"]["num_workers"] = 0
    config["dataset"]["prefetch_factor"] = None

    trainer_name = next(iter(config["trainer"]))
    config["trainer"][trainer_name]["epochs"] = 1

    config_path.write_text(
        yaml.dump(config, sort_keys=False),
        encoding="utf-8",
    )


def _update_lr_sweep(repo_dir: Path) -> None:
    """Limit the scaffolded LR sweep to a single fast run."""
    sweep_path = repo_dir / "experiments" / "lr_sweep.yaml"
    sweep_config = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    sweep_config["grid"] = {"optimizers.lr": [0.001]}
    sweep_path.write_text(
        yaml.dump(sweep_config, sort_keys=False),
        encoding="utf-8",
    )


def test_scaffold_smoke_repo_runs_dl_run_and_dl_sweep() -> None:
    """Create a fresh repo and verify dl-run/dl-sweep wiring end to end."""
    with TemporaryDirectory() as temp_dir:
        root_dir = Path(temp_dir)
        repo_name = "smoke-e2e"
        dataset_name = "smoke_e2e"

        _run_entrypoint(
            "dl-init-experiment",
            "--name",
            repo_name,
            "--root-dir",
            str(root_dir),
            cwd=root_dir,
        )

        repo_dir = root_dir / repo_name
        _write_dummy_dataset_wrapper(repo_dir, dataset_name)
        _update_base_config(repo_dir, dataset_name)
        _update_lr_sweep(repo_dir)

        _run_entrypoint(
            "dl-run",
            "--config",
            "configs/base.yaml",
            cwd=repo_dir,
        )
        _run_entrypoint(
            "dl-sweep",
            "--sweep",
            "experiments/lr_sweep.yaml",
            cwd=repo_dir,
        )
        analyzer_output = _run_entrypoint(
            "dl-analyze",
            "--sweep",
            "experiments/lr_sweep.yaml",
            cwd=repo_dir,
        )

        generated_sweep_dir = repo_dir / "experiments" / "lr_sweep"
        generated_configs = sorted(generated_sweep_dir.glob("*.yaml"))
        assert generated_configs, "Sweep should generate at least one concrete config"
        assert "Tracked runs: 1" in analyzer_output
        assert "status=completed" in analyzer_output

        artifacts_dir = repo_dir / "artifacts"
        assert artifacts_dir.exists(), "Smoke run should create local artifacts"
        assert any(artifacts_dir.rglob("*")), "Artifacts directory should not be empty"

        tracker_path = generated_sweep_dir / "sweep_tracking.json"
        tracker_data = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
        tracked_run = tracker_data["runs"]["0"]
        assert tracked_run["artifact_dir"]
        assert tracked_run["metrics_summary_path"]
        assert tracked_run["metrics_history_path"]

        summary_path = Path(tracked_run["metrics_summary_path"])
        history_path = Path(tracked_run["metrics_history_path"])
        run_info_path = Path(tracked_run["artifact_dir"]) / "final" / "run_info.json"
        assert summary_path.exists()
        assert history_path.exists()
        assert run_info_path.exists()
