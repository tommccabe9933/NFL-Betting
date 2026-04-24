"""Live odds snapshot collector — stub for future implementation.

Saves raw JSON snapshots timestamped under data/raw/odds_snapshots/.
Deployable independently from the historical backfill pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src import paths

logger = logging.getLogger(__name__)


def collect_odds_snapshot(
    api_url: str = None,
    api_key: str = None,
) -> Path:
    """Fetch current odds and save as timestamped JSON.

    This is a stub — no free historical odds API with research quality exists.
    When configured with a real API (e.g. The Odds API), this will:
    1. Call the endpoint for NFL game odds
    2. Save raw JSON with UTC timestamp
    3. Return the path to the saved file

    For now, creates a placeholder to validate the schema.
    """
    if api_url is None:
        logger.info(
            "Odds snapshot collector: no API configured. "
            "Set api_url and api_key to enable live collection."
        )
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = paths.RAW_ODDS_SNAPSHOTS / f"odds_{ts}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import httpx
        with httpx.Client(timeout=30.0) as client:
            headers = {"x-api-key": api_key} if api_key else {}
            resp = client.get(api_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        snapshot = {
            "fetched_at_utc": ts,
            "source": api_url,
            "data": data,
        }
        dest.write_text(json.dumps(snapshot, indent=2))
        logger.info(f"Saved odds snapshot: {dest}")
        return dest

    except Exception as e:
        logger.error(f"Failed to collect odds snapshot: {e}")
        return None
