"""Direct parquet download from nflverse GitHub releases.

Used as fallback when nfl_data_py doesn't support a table (PFR, FTN).
"""

import logging
import time
from pathlib import Path

import httpx

from src.config import NFLVERSE_RELEASE_URL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def download_nflverse_parquet(
    table: str,
    filename: str,
    dest_dir: Path,
    force: bool = False,
) -> Path:
    """Download a parquet file from nflverse GitHub releases.

    Args:
        table: Release tag name (e.g. "pfr_advstats", "ftn_charting")
        filename: File name within the release (e.g. "advstats_season_pass_2023.parquet")
        dest_dir: Directory to save to
        force: Re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    dest = dest_dir / filename
    if dest.exists() and not force:
        logger.debug(f"Already exists, skipping: {dest}")
        return dest

    url = f"{NFLVERSE_RELEASE_URL}/{table}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                resp = client.get(url)
                if resp.status_code == 404:
                    logger.warning(f"Not found (404): {url}")
                    return None
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                logger.info(f"Downloaded {filename} ({len(resp.content) / 1024:.0f} KB)")
                return dest
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"Failed to download {url}: {e}")
                raise
