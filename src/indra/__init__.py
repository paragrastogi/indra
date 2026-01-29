"""Indra package entrypoints."""

from __future__ import annotations

import sys

from . import indra as indra_module


def main(argv: list[str] | None = None) -> int:
    return indra_module.main(argv)


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))
