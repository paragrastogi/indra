"""Indra package entrypoints."""

from __future__ import annotations

import sys

from . import indra as indra_module
from .logging_utils import setup_logger, get_logger


def main(argv: list[str] | None = None, logger=None) -> int:
    logger = logger or setup_logger()
    logger.info("package main start")
    result = indra_module.main(argv, logger=logger)
    logger.info("package main success")
    return result


def cli(logger=None) -> None:
    logger = logger or setup_logger()
    logger.info("package cli start")
    raise SystemExit(main(sys.argv[1:], logger=logger))
