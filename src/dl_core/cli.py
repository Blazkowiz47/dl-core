"""User-facing command line entry point for ``dl-core`` utilities."""

from __future__ import annotations

import argparse

from dl_core.component_scaffold import (
    create_component_scaffold,
    list_supported_component_types,
)


def main(argv: list[str] | None = None) -> int:
    """Run the ``dl-core`` command line interface."""
    parser = argparse.ArgumentParser(
        prog="dl-core",
        description="Utilities for working with dl-core experiment repositories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add",
        help="Create a local component scaffold inside an experiment repository.",
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
        "--root-dir",
        default=".",
        help=(
            "Path inside the target experiment repository. "
            "Defaults to the current directory."
        ),
    )
    add_parser.add_argument(
        "--package-name",
        help=(
            "Explicit local package name to use when the project has multiple "
            "src packages."
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

    component_path = create_component_scaffold(
        component_type=args.component_type,
        name=args.name,
        root_dir=args.root_dir,
        package_name=args.package_name,
        force=args.force,
    )
    print(f"Created component scaffold: {component_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
