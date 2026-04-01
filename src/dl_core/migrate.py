"""Utilities for migrating local ``dl-core`` repository state."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SKIP_TOP_LEVEL_NAMES = {"latest", "runs", "sweeps"}
TRACKING_FILENAMES = ("sweep_tracking.json",)


@dataclass(frozen=True)
class ArtifactMove:
    """One planned artifact-directory migration."""

    source: Path
    destination: Path
    sweep_name: str | None = None


def _is_run_dir(path: Path) -> bool:
    """Return whether ``path`` looks like one run artifact directory."""
    return (
        path.is_dir()
        and (path / "config.yaml").exists()
        and (path / "final").is_dir()
    )


def _resolve_output_dir(root_dir: Path, output_dir: str) -> Path:
    """Resolve the concrete output directory for migration."""
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate
    return root_dir / candidate


def discover_artifact_moves(output_dir: Path) -> list[ArtifactMove]:
    """Discover legacy artifact directories that can be flattened."""
    if not output_dir.exists():
        return []

    moves: list[ArtifactMove] = []
    for experiment_dir in sorted(output_dir.iterdir()):
        if (
            not experiment_dir.is_dir()
            or experiment_dir.is_symlink()
            or experiment_dir.name in SKIP_TOP_LEVEL_NAMES
        ):
            continue

        for child in sorted(experiment_dir.iterdir()):
            if child.is_symlink() or child.name == "latest":
                continue

            if _is_run_dir(child):
                moves.append(
                    ArtifactMove(
                        source=child,
                        destination=output_dir / "runs" / child.name,
                    )
                )
                continue

            if not child.is_dir():
                continue

            for run_dir in sorted(child.iterdir()):
                if run_dir.is_symlink() or run_dir.name == "latest":
                    continue
                if not _is_run_dir(run_dir):
                    continue

                moves.append(
                    ArtifactMove(
                        source=run_dir,
                        destination=output_dir / "sweeps" / child.name / run_dir.name,
                        sweep_name=child.name,
                    )
                )

    return moves


def _find_subsequence(parts: tuple[str, ...], target: tuple[str, ...]) -> int | None:
    """Return the first index where ``target`` appears in ``parts``."""
    if not target or len(target) > len(parts):
        return None

    last_start = len(parts) - len(target) + 1
    for index in range(last_start):
        if parts[index : index + len(target)] == target:
            return index
    return None


def _rewrite_path_string(value: str, source: Path, destination: Path) -> str:
    """Rewrite one string path when it contains the old artifact subpath."""
    value_parts = Path(value).parts
    source_parts = source.parts
    match_index = _find_subsequence(value_parts, source_parts)
    if match_index is None:
        return value

    rewritten_parts = (
        value_parts[:match_index]
        + destination.parts
        + value_parts[match_index + len(source_parts) :]
    )
    return str(Path(*rewritten_parts))


def _rewrite_mapping_paths(
    data: dict[str, Any],
    source: Path,
    destination: Path,
    keys: Iterable[str],
) -> bool:
    """Rewrite selected mapping keys when they point into one run directory."""
    changed = False
    for key in keys:
        value = data.get(key)
        if not isinstance(value, str) or not value:
            continue

        rewritten = _rewrite_path_string(value, source, destination)
        if rewritten == value:
            continue

        data[key] = rewritten
        changed = True

    return changed


def _rewrite_config_metadata(run_dir: Path, source: Path, destination: Path) -> bool:
    """Rewrite ``config.yaml`` artifact metadata after a move."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return False

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    metadata = config.get("run_metadata")
    if not isinstance(metadata, dict):
        return False

    changed = _rewrite_mapping_paths(
        metadata,
        source,
        destination,
        keys=("artifact_dir",),
    )
    if not changed:
        return False

    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
        )

    return True


def _rewrite_run_info(run_dir: Path, source: Path, destination: Path) -> bool:
    """Rewrite ``final/run_info.json`` after a move."""
    run_info_path = run_dir / "final" / "run_info.json"
    if not run_info_path.exists():
        return False

    with open(run_info_path, encoding="utf-8") as handle:
        run_info = json.load(handle)

    changed = _rewrite_mapping_paths(
        run_info,
        source,
        destination,
        keys=(
            "artifact_dir",
            "config_path",
            "metrics_summary_path",
            "metrics_history_path",
        ),
    )
    if not changed:
        return False

    with open(run_info_path, "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2)
        handle.write("\n")

    return True


def _rewrite_sweep_tracking_file(
    tracking_path: Path,
    source: Path,
    destination: Path,
) -> bool:
    """Rewrite matching artifact references inside one ``sweep_tracking.json``."""
    with open(tracking_path, encoding="utf-8") as handle:
        sweep_data = json.load(handle)

    runs = sweep_data.get("runs")
    if not isinstance(runs, dict):
        return False

    changed = False
    for run_data in runs.values():
        if not isinstance(run_data, dict):
            continue
        changed |= _rewrite_mapping_paths(
            run_data,
            source,
            destination,
            keys=("artifact_dir", "metrics_summary_path", "metrics_history_path"),
        )

    if not changed:
        return False

    with open(tracking_path, "w", encoding="utf-8") as handle:
        json.dump(sweep_data, handle, indent=2)
        handle.write("\n")

    return True


def rewrite_metadata(root_dir: Path, source: Path, destination: Path) -> dict[str, int]:
    """Rewrite run-local and sweep-tracker metadata after one move."""
    updated = {
        "config": 0,
        "run_info": 0,
        "sweep_tracking": 0,
    }
    if _rewrite_config_metadata(destination, source, destination):
        updated["config"] += 1
    if _rewrite_run_info(destination, source, destination):
        updated["run_info"] += 1

    for tracking_name in TRACKING_FILENAMES:
        for tracking_path in root_dir.rglob(tracking_name):
            if _rewrite_sweep_tracking_file(tracking_path, source, destination):
                updated["sweep_tracking"] += 1

    return updated


def perform_artifact_migration(
    root_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool,
) -> int:
    """Move legacy artifact directories into the flattened layout."""
    moves = discover_artifact_moves(output_dir)
    if not moves:
        print(f"No legacy artifact directories found under {output_dir}")
        return 0

    conflicts = 0
    total_updates = {
        "config": 0,
        "run_info": 0,
        "sweep_tracking": 0,
    }

    for move in moves:
        print(f"{move.source} -> {move.destination}")
        if move.destination.exists() and move.destination != move.source:
            conflicts += 1
            print("  skipped: destination already exists")
            continue

        if dry_run:
            continue

        move.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.destination))
        updated = rewrite_metadata(root_dir, move.source, move.destination)
        for key, value in updated.items():
            total_updates[key] += value

    print(
        "Summary: "
        f"planned={len(moves)}, conflicts={conflicts}, "
        f"config_updates={total_updates['config']}, "
        f"run_info_updates={total_updates['run_info']}, "
        f"sweep_tracking_updates={total_updates['sweep_tracking']}"
    )
    if dry_run:
        print("Dry run only. No files were moved.")

    return 0 if conflicts == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the ``dl-migrate`` CLI parser."""
    parser = argparse.ArgumentParser(
        prog="dl-migrate",
        description="Migrate local dl-core repository state.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help=(
            "Move legacy artifact directories from "
            "`artifacts/<experiment>/...` into the flattened layout."
        ),
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Experiment repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help=(
            "Artifact output directory relative to root-dir. "
            "Defaults to artifacts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying any files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``dl-migrate`` command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.artifacts:
        parser.error("at least one migration target is required; use --artifacts")

    root_dir = Path(args.root_dir).resolve()
    output_dir = _resolve_output_dir(root_dir, args.output_dir)
    return perform_artifact_migration(
        root_dir,
        output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
