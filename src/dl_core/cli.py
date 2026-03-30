"""User-facing command line entry point for ``dl-core`` utilities."""

from __future__ import annotations

import argparse

from dl_core.component_describer import (
    describe_component,
    format_description,
    format_description_json,
    format_registry_listing,
    format_registry_listing_json,
    list_registered_components,
    list_supported_list_types,
    list_supported_describe_types,
    normalize_list_type,
    normalize_describe_type,
)
from dl_core.component_scaffold import (
    create_component_scaffold,
    list_supported_dataset_bases,
    list_supported_component_types,
    normalize_component_type,
)
from dl_core.sweep_scaffold import (
    create_sweep_scaffold,
    list_supported_tracking_backends,
    normalize_tracking_backend,
)


def _list_supported_add_targets() -> list[str]:
    """Return supported targets for ``dl-core add``."""
    return sorted([*list_supported_component_types(), "sweep"])


def main(argv: list[str] | None = None) -> int:
    """Run the ``dl-core`` command line interface."""
    parser = argparse.ArgumentParser(
        prog="dl-core",
        description="Utilities for working with dl-core experiment repositories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Top-level CLIs:\n"
            "  dl-init-experiment  Initialize an experiment repository scaffold.\n"
            "  dl-core             Add, list, and describe registered types.\n"
            "  dl-run              Run one config locally.\n"
            "  dl-sweep            Expand and execute a sweep config.\n"
            "  dl-analyze          Inspect saved sweep results.\n"
            "  dl-train-worker     Execute one worker config directly.\n\n"
            "Common first steps:\n"
            "  dl-init-experiment --name my-exp --root-dir .\n"
            "  cd my-exp\n"
            "  uv run dl-core list metric_manager\n"
            "  uv run dl-core add dataset LocalDataset\n"
            "  uv run dl-core add sweep DebugSweep\n"
            "  uv run dl-core describe dataset LocalDataset --root-dir .\n"
            "  uv run dl-core add model MyResNet\n"
            "  uv run dl-core add callback MyMetrics\n"
            "  uv run dl-run --config configs/base.yaml\n"
            "  uv run dl-sweep experiments/lr_sweep.yaml\n"
            "  uv run dl-analyze --sweep experiments/lr_sweep.yaml"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add",
        help="Create a local component or sweep scaffold inside an experiment repository.",
        description=(
            "Create a local component or sweep scaffold inside an experiment "
            "repository.\n\nUse this after dl-init-experiment when you want an "
            "extra local dataset, model, callback, trainer, or another sweep "
            "file.\nComponent generation writes the module and updates the "
            "matching src/<package>/__init__.py export list for you."
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
            "Sweep examples:\n"
            "  dl-core add sweep DebugSweep\n"
            "  dl-core add sweep AzureEval --tracking azure_mlflow\n"
            "  dl-core add sweep WandbAblation --tracking wandb\n\n"
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
            f"{', '.join(_list_supported_add_targets())}"
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
        "--tracking",
        default="local",
        help=(
            "Tracking backend to scaffold when component_type is 'sweep'. "
            "Supported values: "
            f"{', '.join(list_supported_tracking_backends())}. "
            "Defaults to local."
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

    describe_parser = subparsers.add_parser(
        "describe",
        help="Describe a registered component class or importable base class.",
        description=(
            "Inspect a registered component or importable class.\n\n"
            "Use this when you want constructor signatures, inheritance, "
            "properties, class attributes, and public methods without opening "
            "the source code first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dl-core describe dataset my_dataset --root-dir .\n"
            "  dl-core describe model tall_swin --root-dir .\n"
            "  dl-core describe class dl_core.core.FrameWrapper\n"
            "  dl-core describe class dl_azure.datasets.AzureComputeMultiFrameWrapper\n"
            "  dl-core describe dataset ocim_multiframe --root-dir . --json"
        ),
    )
    describe_parser.add_argument(
        "component_type",
        help=(
            "Describe target type. Supported values: "
            f"{', '.join(list_supported_describe_types())}"
        ),
    )
    describe_parser.add_argument(
        "name",
        help=(
            "Registered component name or fully qualified class path when "
            "component_type is 'class'."
        ),
    )
    describe_parser.add_argument(
        "--root-dir",
        default=".",
        help=(
            "Path inside the target experiment repository. "
            "Defaults to the current directory."
        ),
    )
    describe_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the description as JSON instead of human-readable text.",
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List registered built-in or local components by type.",
        description=(
            "List registered components for one registry or all registries.\n\n"
            "Use this when you want to see the built-in defaults available in "
            "the current environment before choosing a component name in "
            "config or on the CLI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dl-core list\n"
            "  dl-core list sampler\n"
            "  dl-core list metric_manager --json\n"
            "  dl-core list trainer --root-dir ."
        ),
    )
    list_parser.add_argument(
        "component_type",
        nargs="?",
        default="all",
        help=(
            "Registry target to list. Supported values: "
            f"{', '.join(list_supported_list_types())}. "
            "Defaults to all."
        ),
    )
    list_parser.add_argument(
        "--root-dir",
        default=".",
        help=(
            "Path inside the target experiment repository. "
            "Defaults to the current directory."
        ),
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the registry listing as JSON instead of human-readable text.",
    )

    args = parser.parse_args(argv)
    if args.command == "add":
        normalized_add_target = args.component_type.strip().lower().replace(
            "-", "_"
        ).replace(" ", "_")
        if normalized_add_target == "sweep":
            if args.base:
                parser.error(
                    "--base is only supported when component_type is 'dataset'."
                )
            normalize_tracking_backend(args.tracking)
            sweep_path = create_sweep_scaffold(
                args.name,
                root_dir=args.root_dir,
                tracking_backend=args.tracking,
                force=args.force,
            )
            print(f"Created sweep scaffold: {sweep_path}")
            return 0

        if args.tracking != "local":
            parser.error(
                "--tracking is only supported when component_type is 'sweep'."
            )

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

    if args.command == "describe":
        normalize_describe_type(args.component_type)
        description = describe_component(
            args.component_type,
            args.name,
            root_dir=args.root_dir,
        )
        if args.json:
            print(format_description_json(description))
        else:
            print(format_description(description))
        return 0

    if args.command == "list":
        normalize_list_type(args.component_type)
        listing = list_registered_components(
            args.component_type,
            root_dir=args.root_dir,
        )
        if args.json:
            print(format_registry_listing_json(listing))
        else:
            print(format_registry_listing(listing))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
