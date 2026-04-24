"""As-of join framework for timestamped sources.

Backward joins only: for each row in left, find the most recent row in right
where right.ts <= left.ts. Used for injuries, depth charts, odds snapshots.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def as_of_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: list,
    left_time: str,
    right_time: str,
    direction: str = "backward",
) -> pl.DataFrame:
    """Generic backward as-of join.

    For each row in left, joins the most recent row from right where
    right[right_time] <= left[left_time], matching on `on` columns.

    Args:
        left: Base table (e.g., games with kickoff times)
        right: Timestamped source (e.g., injury reports with date_modified)
        on: Exact-match columns (e.g., ["team", "season"])
        left_time: Timestamp column in left (e.g., "kickoff_ts_utc")
        right_time: Timestamp column in right (e.g., "date_modified")
        direction: "backward" (default) — only join past records

    Returns:
        Left table with right columns joined.
    """
    if direction != "backward":
        raise ValueError("Only backward as-of joins are supported (leakage safety)")

    # Sort both by time
    left_sorted = left.sort(left_time)
    right_sorted = right.sort(right_time)

    # Polars join_asof
    result = left_sorted.join_asof(
        right_sorted,
        left_on=left_time,
        right_on=right_time,
        by=on if on else None,
        strategy="backward",
    )

    # Validate: no right timestamps exceed left timestamps
    if right_time in result.columns and left_time in result.columns:
        violations = result.filter(
            pl.col(right_time) > pl.col(left_time)
        )
        if len(violations) > 0:
            logger.error(
                f"LEAKAGE DETECTED: {len(violations)} rows have "
                f"{right_time} > {left_time}"
            )
            raise ValueError("As-of join produced future data — leakage detected")

    return result
