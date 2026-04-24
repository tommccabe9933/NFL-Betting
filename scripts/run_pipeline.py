#!/usr/bin/env python3
"""NFL Pregame Feature Pipeline — CLI runner.

Usage:
    python scripts/run_pipeline.py                    # Full pipeline, all seasons
    python scripts/run_pipeline.py --seasons 2023 2024  # Specific seasons only
    python scripts/run_pipeline.py --phase ingest       # Only run ingestion
    python scripts/run_pipeline.py --phase features     # Only run feature builders
    python scripts/run_pipeline.py --phase assemble     # Only run assembly
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, paths
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_ingest(seasons: list):
    """Phase 1: Raw data ingestion."""
    logger.info(f"=== PHASE 1: INGESTION (seasons {seasons[0]}-{seasons[-1]}) ===")

    from src.data_ingest.market import ingest_market
    from src.data_ingest.schedules import ingest_schedules
    from src.data_ingest.pbp import ingest_pbp
    from src.data_ingest.player_stats import ingest_player_stats
    from src.data_ingest.injuries import ingest_injuries
    from src.data_ingest.depth_charts import ingest_depth_charts
    from src.data_ingest.ngs import ingest_ngs
    from src.data_ingest.pfr import ingest_pfr
    from src.data_ingest.ftn import ingest_ftn

    ingest_market()
    ingest_schedules(seasons)
    ingest_pbp(seasons)
    ingest_player_stats(seasons)
    ingest_injuries(seasons)
    ingest_depth_charts(seasons)

    ngs_seasons = [s for s in seasons if s >= config.NGS_MIN_SEASON]
    if ngs_seasons:
        ingest_ngs(ngs_seasons)

    pfr_seasons = [s for s in seasons if s >= config.PFR_MIN_SEASON]
    if pfr_seasons:
        ingest_pfr(pfr_seasons)

    ftn_seasons = [s for s in seasons if s >= config.FTN_MIN_SEASON]
    if ftn_seasons:
        ingest_ftn(ftn_seasons)


def run_normalize():
    """Phase 2: Normalization."""
    logger.info("=== PHASE 2: NORMALIZATION ===")

    from src.normalize.games import build_and_save
    build_and_save()


def run_features():
    """Phase 3: Feature builders."""
    logger.info("=== PHASE 3: FEATURE BUILDERS ===")

    from src.features.market_baseline import build_market_features
    from src.features.epa_ratings import build_epa_ratings
    from src.features.elo import build_elo_ratings
    from src.features.rest_schedule import build_rest_features
    from src.features.weather import build_weather_features
    from src.features.qb_features import build_qb_features
    from src.features.injury_impact import build_injury_features
    from src.features.situational_epa import build_situational_epa
    from src.features.tracking_proxy import build_tracking_proxy_features
    from src.features.pfr_pressure import build_pfr_features
    from src.features.ftn_scheme import build_ftn_features
    from src.features.kaggle_team_stats import build_kaggle_team_features

    build_market_features()
    build_epa_ratings()
    build_elo_ratings()
    build_rest_features()
    build_weather_features()
    build_qb_features()
    build_injury_features()
    build_situational_epa()
    build_tracking_proxy_features()
    build_pfr_features()
    build_ftn_features()
    build_kaggle_team_features()


def run_assemble():
    """Phase 4-5: Assembly."""
    logger.info("=== PHASE 4-5: ASSEMBLY ===")

    from src.assembly.feature_store import assemble_feature_store
    result = assemble_feature_store()

    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE STORE COMPLETE")
    logger.info(f"  Games: {len(result)}")
    logger.info(f"  Columns: {len(result.columns)}")
    logger.info(f"  Seasons: {result['season'].min()}-{result['season'].max()}")
    logger.info(f"  Output: {paths.MODELING / 'feature_store_v1.parquet'}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="NFL Pregame Feature Pipeline")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=None,
        help="Seasons to process (default: all 2012-2025)",
    )
    parser.add_argument(
        "--phase", choices=["ingest", "normalize", "features", "assemble", "all"],
        default="all", help="Which phase to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    seasons = args.seasons or list(range(config.MIN_SEASON, config.MAX_SEASON + 1))

    paths.ensure_dirs()
    start = time.time()

    if args.phase in ("ingest", "all"):
        run_ingest(seasons)
    if args.phase in ("normalize", "all"):
        run_normalize()
    if args.phase in ("features", "all"):
        run_features()
    if args.phase in ("assemble", "all"):
        run_assemble()

    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
