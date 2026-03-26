"""User-facing command line entry point for ``dl-core`` utilities."""

from __future__ import annotations

import argparse

from dl_core.component_scaffold import (
    create_component_scaffold,
    list_supported_dataset_bases,
    list_supported_component_types,
    normalize_component_type,
)


def main(argv: list[str] | None = None) -> int:
    """Run the ``dl-core`` command line interface."""
    parser = argparse.ArgumentParser(
        prog="dl-core",
        description="Utilities for working with dl-core experiment repositories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common first steps:\n"
            "  dl-init-experiment --name my-exp --root-dir .\n"
            "  cd my-exp\n"
            "  uv run dl-core add dataset LocalDataset\n"
            "  uv run dl-core add model MyResNet\n"
            "  uv run dl-core add callback MyMetrics\n"
            "  uv run dl-run --config configs/base.yaml"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add",
        help="Create a local component scaffold inside an experiment repository.",
        description=(
            "Create a local component scaffold inside an experiment repository.\n\n"
            "Use this after dl-init-experiment when you want an extra local "
            "dataset, model, callback, trainer, or other component.\n"
            "The command writes the component module and updates the matching "
            "src/<package>/__init__.py export list for you."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Dataset examples:\n"
            "  dl-core add dataset LocalDataset\n"
            "  dl-core add dataset FrameDataset --base frame\n"
            "  dl-core add dataset TextDataset --base text_sequence\n"
            "  dl-core add dataset ActDataset --base adaptive_computation\n"
            "  dl-core add dataset AzureFrames --base azure_compute_frame\n"
            "  dl-core add dataset AzureSeq --base azure_compute_multiframe\n\n"
            "Other component examples:\n"
            "  dl-core add model MyResNet\n"
            "  dl-core add trainer MyTrainer\n"
            "  dl-core add callback MyMetrics\n"
            "  dl-core add metric_manager MyManager\n"
            "  dl-core add sampler MySampler\n"
            "  dl-core add criterion MyLoss\n"
            "  dl-core add augmentation MyAugmentation\n"
            "  dl-core add metric MyMetric\n"
            "  dl-core add executor MyExecutor"
        ),
    )
    add_parser.add_argument(
        "component_type",
        help=(
            "Component type to generate. Supported values: "
            f"{', '.join(list_supported_component_types())}"
        ),
    )
    add_parser.add_argument(
        "name",
        help="Display name for the new component. It will also be normalized.",
    )
    add_parser.add_argument(
        "--base",
        help=(
            "Dataset scaffold base to use when component_type is 'dataset'. "
            "Supported values in this environment: "
            f"{', '.join(list_supported_dataset_bases())}."
        ),
    )
    add_parser.add_argument(
        "--root-dir",
        default=".",
        help=(
            "Path inside the target experiment repository. "
            "Defaults to the current directory."
        ),
    )
    add_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing component scaffold if it already exists.",
    )

    args = parser.parse_args(argv)
    if args.command != "add":
        parser.error(f"Unsupported command: {args.command}")
    if args.base and normalize_component_type(args.component_type) != "dataset":
        parser.error("--base is only supported when component_type is 'dataset'.")

    component_path = create_component_scaffold(
        component_type=args.component_type,
        name=args.name,
        root_dir=args.root_dir,
        dataset_base=args.base,
        force=args.force,
    )
    print(f"Created component scaffold: {component_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
