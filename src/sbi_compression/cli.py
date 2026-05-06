"""Command line interface for compression workflows."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbi_compression_cli",
        description="Run compression workflows for cosmological data.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="sbi-compression 0.1.0",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
