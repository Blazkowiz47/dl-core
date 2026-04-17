"""Inspect dataset split sizes and one example batch from a config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from dl_core import load_builtin_components, load_local_components
from dl_core.core import DATASET_REGISTRY
from dl_core.utils.config import load_config

_SPLITS = ("train", "validation", "test")


def _force_local_loader_settings(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """Force single-process DataLoader settings for local inspection."""
    config = dict(dataset_config)
    config["num_workers"] = {split: 0 for split in _SPLITS}
    config["prefetch_factor"] = {split: None for split in _SPLITS}
    return config


def _format_value(value: Any) -> str:
    """Format a batch value for readable inspection output."""
    if isinstance(value, torch.Tensor):
        return f"tensor(shape={list(value.shape)}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return repr(value)


def _print_mapping(title: str, values: dict[str, Any]) -> None:
    """Print a flat mapping with formatted values."""
    print(title)
    for key, value in values.items():
        print(f"  - {key}: {_format_value(value)}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for dataset inspection."""
    parser = argparse.ArgumentParser(
        prog="dl-inspect-dataset",
        description="Inspect dataset split sizes and one collated batch.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--sample-split",
        default="train",
        choices=list(_SPLITS),
        help="Split to preview after computing split sizes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run dataset inspection for one config."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    load_builtin_components()
    load_local_components(config_path)
    config = load_config(str(config_path))

    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict) or not dataset_config.get("name"):
        parser.error("dataset.name is required for dl-inspect-dataset")
    dataset_config = _force_local_loader_settings(dataset_config)
    dataset_config.setdefault("deterministic", config.get("deterministic", True))

    dataset_name = str(dataset_config["name"])
    dataset = DATASET_REGISTRY.get(
        dataset_name,
        dataset_config,
    )

    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_name}")
    if getattr(dataset, "classes", None):
        print(f"Classes: {list(dataset.classes)}")

    print("Split sizes")
    preview_loader = None
    for split in _SPLITS:
        dataloader = dataset.get_split(split)
        if dataloader is None:
            print(f"  - {split}: unavailable")
            continue

        split_size = len(dataloader.dataset)
        print(f"  - {split}: {split_size} samples")
        if split == args.sample_split:
            preview_loader = dataloader

    if preview_loader is None or len(preview_loader.dataset) == 0:
        print(f"No preview batch available for split: {args.sample_split}")
        return 0

    batch = next(iter(preview_loader))
    _print_mapping(f"{args.sample_split} batch", batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
