"""Helpers for generating experiment sweep scaffolds."""

from __future__ import annotations

import re
from pathlib import Path

from dl_core.project import find_project_root

_TRACKING_BACKENDS = {
    "local": "  backend: local\n",
    "mlflow": "  backend: mlflow\n  tracking_uri: ./mlruns\n",
    "wandb": "  backend: wandb\n  project: my_experiment\n  entity: null\n",
    "azure_mlflow": "  backend: azure_mlflow\n",
}

_SWEEP_NAME_COMMENT = (
    "  # sweep_name: custom_sweep_name\n"
    "  # Optional override. Defaults to the sweep filename.\n"
)


def list_supported_tracking_backends() -> list[str]:
    """Return supported tracking backend names for sweep scaffolds."""
    return sorted(_TRACKING_BACKENDS.keys())


def normalize_tracking_backend(tracking_backend: str | None) -> str:
    """Normalize a user-provided tracking backend string."""
    if tracking_backend is None:
        return "local"

    key = tracking_backend.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _TRACKING_BACKENDS:
        supported = ", ".join(list_supported_tracking_backends())
        raise ValueError(
            f"Unsupported tracking backend '{tracking_backend}'. Supported backends: "
            f"{supported}"
        )
    return key


def create_sweep_scaffold(
    name: str,
    *,
    root_dir: str = ".",
    tracking_backend: str = "local",
    force: bool = False,
) -> Path:
    """Create a new sweep file inside an experiment repository."""
    search_root = Path(root_dir).resolve()
    project_root = find_project_root(search_root)
    if project_root is None:
        raise FileNotFoundError(
            "Could not find an experiment repository from the provided root. "
            "Run this inside a repository created by dl-init-experiment or pass "
            "--root-dir."
        )

    experiments_dir = project_root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    sweep_name = _to_module_name(name)
    normalized_tracking = normalize_tracking_backend(tracking_backend)
    sweep_path = experiments_dir / f"{sweep_name}.yaml"

    if sweep_path.exists() and not force:
        raise FileExistsError(f"Sweep already exists: {sweep_path}")

    sweep_path.write_text(
        _render_sweep(sweep_name, normalized_tracking),
        encoding="utf-8",
    )
    return sweep_path


def _render_sweep(sweep_name: str, tracking_backend: str) -> str:
    """Render a sweep scaffold with an empty grid."""
    description = sweep_name.replace("_", " ")
    return (
        f"# Sweep: {sweep_name}\n\n"
        'extends_template: "../configs/base_sweep.yaml"\n'
        f'description: "{description}"\n\n'
        "fixed:\n"
        "  accelerators: preset:accelerators.cpu\n"
        "  executors: preset:executors.local\n\n"
        "grid: {}\n\n"
        "tracking:\n"
        f"{_SWEEP_NAME_COMMENT}"
        f"{_TRACKING_BACKENDS[tracking_backend]}"
    )


def _slugify(value: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    if not normalized:
        raise ValueError("Name must contain at least one alphanumeric character")
    return normalized


def _to_module_name(value: str) -> str:
    """Convert text to a Python-style module name."""
    module_name = _slugify(value).replace("-", "_")
    if module_name[0].isdigit():
        module_name = f"exp_{module_name}"
    return module_name
