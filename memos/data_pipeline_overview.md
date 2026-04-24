# NFL Pregame Feature Pipeline: What We're Building and Why

## The Project So Far

This project started as a STAT 301-2 final exploring whether publicly available NFL team performance data can predict spread coverage beyond what the betting market already prices in. The original R pipeline built a 30-feature model across 3,538 regular season games (2012-2024), tested eight model types, and arrived at an honest conclusion: with proper temporal validation, the model's Brier score is 0.251 against a coin-flip baseline of 0.250. The market is extremely efficient.

That finding stands. But the methodology exposed a clear gap: the feature set was narrow. The original model relied on team-level rolling averages of EPA, yards per play, and basic game context. It lacked player-level signals, defensive scheme data, tracking metrics, and structured injury information — all things that NFL analytics departments and sophisticated bettors use. The question shifted from "can we beat the market with basic stats" to "what is the minimum viable data infrastructure needed to seriously attempt this, and what does a properly engineered feature store look like?"

## What the Python Pipeline Does

We are building a modular, Python-first data pipeline that ingests, normalizes, and transforms NFL data from multiple public sources into a single pregame feature store. The pipeline replaces the monolithic R script approach with a structured engineering system designed for reproducibility, testability, and future expansion.

The pipeline has five phases:

**Phase 1 — Raw Ingestion.** Ten separate ingestion modules pull data from nflverse (the standard open NFL data repository) and existing Kaggle datasets. Each source is downloaded as Parquet files partitioned by season. Play-by-play data (the largest table at ~400 columns per season) is pruned to only the 20 columns needed for feature engineering. Next Gen Stats, PFR advanced stats, FTN charting data, injury reports, and depth charts each have dedicated ingest modules. An odds snapshot collector is stubbed for future live line tracking.

**Phase 2 — Normalization.** Team abbreviations are standardized across all sources using the same six-rule mapping the R pipeline established (handling franchise relocations like Oakland to Las Vegas, San Diego to LA Chargers, St. Louis to LA Rams). A canonical games table is built: one row per game with season, week, teams, spread, total, moneylines, weather, and game context. This table becomes the spine that all feature families join to.

**Phase 3 — Feature Builders.** Twelve independent feature modules each produce a Parquet file in the feature store. Every rolling metric uses a lagged window — at game G, the feature reflects games 1 through G-1, never game G itself. This is the core leakage-safety guarantee. The feature families are:

- Market baseline: spread, total, moneylines, de-vigged implied probabilities
- EPA team ratings: offensive and defensive EPA per play, pass/rush splits, success rate, CPOE (5-game and 3-game rolling)
- Elo ratings: computed sequentially with K=20 and 25% season regression toward the mean
- QB features: rolling QB EPA, completion probability, sack rate, plus injury/depth chart-derived availability flags
- Rest and schedule: days of rest, short week, divisional, primetime, dome, turf, neutral site
- Weather: temperature, wind bins, wind hinge at 12 mph, cold game indicator
- Injury impact: positional-weighted injury scores (a QB listed Out contributes 10 points; a punter Out contributes 1)
- Situational EPA: 3rd down, redzone, early down efficiency, explosive and negative play rates
- Tracking proxies: Next Gen Stats metrics (time to throw, aggressiveness, rush yards over expected)
- PFR pressure: pressure rate, drop rate, bad throw rate (available 2018+)
- FTN scheme: play action rate, RPO rate, blitz rate (available 2022+, feature-flagged)
- Kaggle operational stats: yards per play, pass/rush efficiency from the existing CSV sources

**Phase 4 — As-of Join Framework.** A generic backward as-of join utility ensures that timestamped sources (injuries, depth charts, future odds snapshots) are joined using only records that precede the game's kickoff time. The framework includes an assertion that throws an error if any source timestamp exceeds the bet placement time.

**Phase 5 — Feature Store Assembly.** A final assembly module joins all feature families into one table. For each game, features are joined separately for the favorite and underdog, then differentials are computed (favorite minus underdog). Coverage flags indicate which advanced feature families are available per season (PFR from 2018, NGS from 2016, FTN from 2022). The output is a single Parquet file with one row per game and 80+ columns, ready for modeling.

## Why This Matters

The original R pipeline proved the methodology works but lacked the data depth to find a real edge. The Python pipeline addresses three specific weaknesses:

**Data breadth.** The R pipeline had 30 features. The Python pipeline produces 80+, including player-level QB metrics, defensive scheme indicators, and tracking-derived efficiency measures that were not previously available. The permutation validation from the R pipeline showed that NGS was the only feature block with statistically significant incremental signal — this pipeline makes it easy to add and validate new feature blocks without restructuring the entire codebase.

**Leakage safety at scale.** Every feature builder uses the same lagged rolling utility, and the test suite includes 28 tests specifically verifying that no feature uses current-game or future-game data. The as-of join framework enforces this constraint for timestamped sources. This is not a convenience — it is the difference between a model that appears to work and one that actually works.

**Modularity for iteration.** Each data source and feature family is an independent module. If a new data source becomes available (historical line movement, weather forecasts, player tracking), it can be added without touching existing code. The feature-flag system lets experimental features (FTN charting) be toggled on and off. The CLI runner supports running individual pipeline phases or specific seasons, so a full rebuild is not required to test a change.

## Current State and Next Steps

The pipeline code is complete: 42 Python files across ingestion, normalization, features, assembly, and tests. All 28 unit tests pass, covering team normalization, rolling window leakage safety, and Elo computation correctness. The pipeline has not yet been run against live data — the next step is to install the nfl_data_py package and execute `python scripts/run_pipeline.py --seasons 2023 2024` to produce the first feature store.

Once the feature store is built, the modeling phase can begin using the temporal walk-forward validation framework already established in the R pipeline. The honest expectation is that the market remains very efficient and most new features will not survive permutation validation. But the infrastructure is now in place to test that hypothesis rigorously rather than hopefully.
