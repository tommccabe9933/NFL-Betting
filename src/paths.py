"""Path constants for the NFL pregame feature pipeline."""

from pathlib import Path

# Project root — walk up from this file to find .git
_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parent  # src/ is one level below root

# Raw data directories
RAW = PROJECT_ROOT / "data" / "raw"
RAW_PBP = RAW / "pbp"
RAW_SCHEDULES = RAW / "schedules"
RAW_PLAYER_STATS = RAW / "player_stats"
RAW_NGS = RAW / "ngs"
RAW_PFR = RAW / "pfr"
RAW_FTN = RAW / "ftn"
RAW_INJURIES = RAW / "injuries"
RAW_DEPTH_CHARTS = RAW / "depth_charts"
RAW_MARKET = RAW / "market"
RAW_PARTICIPATION = RAW / "participation"
RAW_ODDS_SNAPSHOTS = RAW / "odds_snapshots"
RAW_WEATHER = RAW / "weather"

# Processed data directories
PROCESSED = PROJECT_ROOT / "data" / "processed"
NORMALIZED = PROCESSED / "normalized"
FEATURE_STORE = PROCESSED / "feature_store"
MODELING = PROCESSED / "modeling"

# Legacy data (existing Kaggle CSVs)
LEGACY_DATA = PROJECT_ROOT / "data"

# Existing R objects for validation
OBJECTS = PROJECT_ROOT / "objects"


def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [
        RAW_PBP, RAW_SCHEDULES, RAW_PLAYER_STATS, RAW_NGS, RAW_PFR,
        RAW_FTN, RAW_INJURIES, RAW_DEPTH_CHARTS, RAW_MARKET,
        RAW_PARTICIPATION, RAW_ODDS_SNAPSHOTS, RAW_WEATHER,
        NORMALIZED, FEATURE_STORE, MODELING,
    ]:
        d.mkdir(parents=True, exist_ok=True)
