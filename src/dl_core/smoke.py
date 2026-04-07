"""Lightweight one-batch smoke checks for experiment configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from dl_core import load_builtin_components, load_local_components
from dl_core.core import DATASET_REGISTRY, MODEL_REGISTRY
from dl_core.utils.config import load_config


def _force_local_loader_settings(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """Force single-process DataLoader settings for a quick smoke check."""
    config = dict(dataset_config)
    config["num_workers"] = {"train": 0, "validation": 0, "test": 0}
    config["auto_split"] = False
    config["validation_partition"] = 0.0
    config["test_split"] = 0.0
    config["prefetch_factor"] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    return config


def _resolve_first_model(
    models_config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Resolve the first configured model block."""
    if not isinstance(models_config, dict) or not models_config:
        raise ValueError("At least one model must be configured for dl-smoke")

    model_key, model_config = next(iter(models_config.items()))
    if not isinstance(model_config, dict):
        raise ValueError(f"models.{model_key} must be a mapping")

    resolved_config = dict(model_config)
    model_name = str(resolved_config.pop("name", model_key))
    return str(model_key), model_name, resolved_config


def _format_value(value: Any) -> str:
    """Format a value for readable smoke output."""
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
    """Create the CLI parser for the smoke helper."""
    parser = argparse.ArgumentParser(
        prog="dl-smoke",
        description="Inspect one dataset batch and one model forward pass.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to inspect.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the lightweight smoke helper."""
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
        parser.error("dataset.name is required for dl-smoke")
    dataset_name = str(dataset_config["name"])
    dataset = DATASET_REGISTRY.get(
        dataset_name,
        _force_local_loader_settings(dataset_config),
    )
    dataloader = dataset.get_split(args.split)
    if dataloader is None:
        parser.error(f"No DataLoader available for split: {args.split}")

    batch = next(iter(dataloader))
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    _print_mapping("Batch", batch)

    try:
        model_key, model_name, model_config = _resolve_first_model(
            config.get("models", {})
        )
    except ValueError as exc:
        parser.error(str(exc))

    model = MODEL_REGISTRY.get(model_name, model_config)
    model.eval()

    with torch.no_grad():
        output = model(batch)

    print(f"Model: {model_key} -> {model_name}")
    if isinstance(output, dict):
        _print_mapping("Forward output", output)
    else:
        print(f"Forward output: {_format_value(output)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
