"""Compatibility stub for future ``dl-core`` repository migrations."""

from __future__ import annotations

import argparse


_MIGRATION_MESSAGE = (
    "No built-in migrations are currently active.\n"
    "\n"
    "The `dl-migrate` command is reserved for future major repository or "
    "artifact layout changes. For now it only serves as a compatibility "
    "placeholder so older guidance or scripts do not fail unexpectedly."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``dl-migrate`` CLI parser."""
    parser = argparse.ArgumentParser(
        prog="dl-migrate",
        description=(
            "Compatibility placeholder for future dl-core repository migrations."
        ),
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help=(
            "Reserved artifact-migration target. Currently inactive and kept "
            "only for forward compatibility."
        ),
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help=(
            "Repository root placeholder. Currently ignored because no "
            "migrations are active."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help=(
            "Artifact output directory placeholder. Currently ignored because "
            "no migrations are active."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Compatibility flag. The command is already a no-op until future "
            "migrations are introduced."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``dl-migrate`` command line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    print(_MIGRATION_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
