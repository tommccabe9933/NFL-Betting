"""I/O utilities for parquet read/write with logging."""

import logging
from pathlib import Path
from typing import Optional, List

import polars as pl

logger = logging.getLogger(__name__)


def write_parquet(df: pl.DataFrame, path: Path, **kwargs) -> None:
    """Write a Polars DataFrame to parquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, **kwargs)
    logger.info(f"Wrote {len(df)} rows to {path}")


def read_parquet(
    path: Path,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Read a parquet file, optionally selecting columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pl.read_parquet(path, columns=columns)
    logger.info(f"Read {len(df)} rows from {path}")
    return df


def scan_parquet(
    path: Path,
    columns: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """Lazy-scan a parquet file for memory-efficient processing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    lf = pl.scan_parquet(path)
    if columns:
        lf = lf.select(columns)
    return lf


def read_parquet_dir(
    dir_path: Path,
    pattern: str = "*.parquet",
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Read all parquet files from a directory into one DataFrame."""
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files in {dir_path}")
    frames = []
    for f in files:
        df = pl.read_parquet(f, columns=columns)
        frames.append(df)
    try:
        combined = pl.concat(frames, how="diagonal_relaxed")
    except Exception:
        combined = pl.concat(frames, how="align")
    logger.info(f"Read {len(combined)} rows from {len(files)} files in {dir_path}")
    return combined
