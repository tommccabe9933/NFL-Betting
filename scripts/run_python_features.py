#!/usr/bin/env python3
"""Rebuild the Python feature store from the raw files already on disk.

Used by run_weekly.sh. Skips nfl_data_py ingestion: 02_prepare_recent_data.R
downloads the current-season raw files. Run with /usr/bin/python3 (it has
polars and nfl_data_py installed).
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import setup_logging
from src.data_ingest.market import ingest_market
from src.normalize.games import build_and_save
from src.assembly.feature_store import assemble_feature_store

setup_logging()
ingest_market(force=True)
build_and_save()
for mod, fn in [
    ("market_baseline", "build_market_features"), ("epa_ratings", "build_epa_ratings"),
    ("elo", "build_elo_ratings"), ("rest_schedule", "build_rest_features"),
    ("weather", "build_weather_features"), ("qb_features", "build_qb_features"),
    ("injury_impact", "build_injury_features"), ("situational_epa", "build_situational_epa"),
    ("tracking_proxy", "build_tracking_proxy_features"),
]:
    getattr(importlib.import_module(f"src.features.{mod}"), fn)()
store = assemble_feature_store()
print(f"Feature store: {store.shape[0]} games x {store.shape[1]} columns")
