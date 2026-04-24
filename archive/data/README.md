# Archived Data

Files not used by the v3.5 final pipeline.

| File | Size | Why archived |
|------|------|--------------|
| `nfl_2025.csv` | 32 KB | Standalone 2025 schedule snapshot — superseded by `data/raw/schedules/schedules_2025.parquet` from the Python pipeline |
| `weekly_player_stats_defense.csv` | 63 MB | Player-level weekly defense stats from initial Kaggle ingest — never integrated into the modeling table. Game-level defense aggregates come from `data/weekly_team_stats_defense.csv` instead. **Gitignored due to size** — present locally but not tracked in version control |

These files came from the original Kaggle data sources used in the initial project phase (see `archive/reports/`). The v3.5 pipeline replaces them with the Python-ingested parquet feature store under `data/raw/` and `data/processed/`.
