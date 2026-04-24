"""Player ID normalization using gsis_id as canonical identifier."""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from src.normalize.teams import normalize_team_expr

logger = logging.getLogger(__name__)


def build_player_mapping(rosters_dir: Path) -> pl.DataFrame:
    """Build canonical player ID mapping from roster files.

    Returns DataFrame with columns: gsis_id, player_name, position, team
    Uses the most recent roster entry per player.
    """
    files = sorted(rosters_dir.glob("*.parquet"))
    if not files:
        logger.warning(f"No roster files in {rosters_dir}")
        return pl.DataFrame(schema={
            "gsis_id": pl.Utf8,
            "player_name": pl.Utf8,
            "position": pl.Utf8,
            "team": pl.Utf8,
        })

    frames = [pl.read_parquet(f) for f in files]
    rosters = pl.concat(frames)

    # Keep most recent entry per player
    mapping = (
        rosters
        .filter(pl.col("gsis_id").is_not_null())
        .with_columns(normalize_team_expr("team"))
        .sort("season", descending=True)
        .group_by("gsis_id")
        .first()
        .select("gsis_id", "player_name", "position", "team")
    )

    logger.info(f"Built player mapping: {len(mapping)} players")
    return mapping
