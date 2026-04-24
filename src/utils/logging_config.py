"""Logging configuration."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
