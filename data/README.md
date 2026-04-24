# Data

This directory holds all data used by the v3.5 pipeline. There are two layers:

1. **Legacy CSVs** at the root — used by `scripts/01_build_model_data.R` to build the initial team-game table
2. **Python pipeline data** under `raw/` and `processed/` — used by `scripts/19b_merge_python_features.R` and `scripts/19c_rebuild_v3_clean.R` to produce the final modeling table

## Active Files

### Legacy CSV ingest (Kaggle)
| File | Used by | Source |
|------|---------|--------|
| `nfl_all_seasons.csv` | `01_build_model_data.R` | [Spreadspoke (Kaggle)](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data) |
| `weekly_team_stats_offense.csv` | `01_build_model_data.R` | [NFL Team Data 2003-2023 (Kaggle)](https://www.kaggle.com/datasets/nickcantalupa/nfl-team-data-2003-2023) |
| `weekly_team_stats_defense.csv` | `01_build_model_data.R` | same as above |

### Python pipeline raw downloads (`data/raw/`)
113 parquet files across 12 sources (`pbp/`, `ngs/`, `ftn/`, `pfr/`, `injuries/`, `depth_charts/`, `schedules/`, `player_stats/`, `weather/`, `market/`, `odds_snapshots/`, `participation/`). Downloaded by `src/data_ingest/*.py` modules. Run `python scripts/run_pipeline.py` to refresh.

### Python pipeline processed outputs (`data/processed/`)
- `feature_store/` — 10 parquet files, one per feature family (Elo, EPA, QB, weather, injury, FTN, market, situational, rest, tracking)
- `modeling/feature_store_v1.parquet` — joined feature store (input to R-side merge)
- `normalized/canonical_games.parquet` — canonical game-id normalized index

## Final Modeling Output

The final modeling table is `objects/modeling_data_v3_FINAL.rds` — **3,538 games × 300 columns**, built by `scripts/19c_rebuild_v3_clean.R`. It contains the active stable feature set (229 features) plus the target (`fav_cover`), market columns, and metadata.

See `memos/feature_summary.md` for the full feature codebook.

## Archived Data

Files not used by the v3.5 pipeline live in `archive/data/`:
- `nfl_2025.csv` — superseded by Python-ingested `schedules_2025.parquet`
- `weekly_player_stats_defense.csv` (63MB, gitignored) — never integrated; aggregates come from team-level CSV instead
