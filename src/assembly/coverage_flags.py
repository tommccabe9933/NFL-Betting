"""Coverage flags — which feature families are available per game."""

from src import config


def has_pfr(season: int) -> bool:
    return season >= config.PFR_MIN_SEASON


def has_ngs(season: int) -> bool:
    return season >= config.NGS_MIN_SEASON


def has_ftn(season: int) -> bool:
    return config.FTN_ENABLED and season >= config.FTN_MIN_SEASON


def has_participation(season: int) -> bool:
    return season >= config.PARTICIPATION_MIN_SEASON
