# Predicting NFL Point Spread Coverage — V3.5 Final

**STAT 301-2: Data Science 2 — Final Project**
**Author:** Thomas McCabe

## Final Result

A combined cover + no-cover ensemble achieves **56.0% against the spread** over **520 out-of-sample bets** across 2023–2025. Bootstrap 90% CI = [52.5%, 59.4%]; entire interval clears the 52.4% breakeven threshold. The 2025 holdout (never seen during development) was profitable: 54.9%, +$860.

**v3.7 (Oct 2026): fav/dog bug fixed, and no edge.** `src/normalize/games.py` now reads nflverse `spread_line` the right way round (positive = home favored), so every `fav_*`/`dog_*` feature describes the right team. Composite scaling is learned on 2012–2022 only. Re-running the identical strategy with nothing re-tuned gives 2023 57.6%, 2024 53.5%, **2025 49.1%**: 430 bets at 54.0%, 90% CI [49.8%, 57.7%], p = 0.28. Calibration finds the model's probabilities carry almost no information (cover slope 0.05, no-cover slope −0.04). The earlier edge does not survive correct data. Treat 2026 picks (`MODEL_VERSION=v37`, the default) as paper trades only.

**v3.6 rebuild (Oct 2026):** The v3.5 table (`modeling_data_v3_FINAL.rds`) had 145 of its ~300 features blank for every 2025 game, because the Kaggle team stats and the Python play-by-play features stopped at 2024. The model filled those blanks with training medians, so the 54.9% "2025 holdout" tested a partial model. A complete-data version was tried in April, scored worse, and was set aside, so 2025 was not a clean blind test either. With every feature rebuilt (`objects/modeling_data_v36.rds`), the same strategy goes **51.7% on 151 bets in 2025 (−$230)**. That is below the 52.4% breakeven. Over 2023–2025 it goes 55.0% on 507 bets: 90% CI [51.3%, 58.6%], binomial p = 0.13. Known issue, kept so results stay comparable: the Python feature store reads nflverse `spread_line` backwards, so its `fav_*`/`dog_*` columns describe the opposite team. Only their `*_diff` columns are corrected. Weekly 2026 picks: `scripts/47_weekly_picks.R`.

**Bet sizing (revised Oct 2026):** the model's raw probabilities are overconfident, so Kelly sizing off them overbets. After calibration (`scripts/45_calibrated_kelly.R`), the recommendation is a flat 1% of bankroll per bet: $10,000 → $13,932 (+39%) over 2023–2025, worst drawdown 9%.

Read the full story: **`memos/blog_post.md`** or open the interactive exhibit: **`exhibit_beating_the_spread.html`**.

## Active Repository Structure

```
├── scripts/                              # Active v3.5 pipeline (run in order)
│   ├── 00_project_setup.R                # Shared setup
│   ├── 01_build_model_data.R             # R-side data ingest + feature engineering
│   ├── 19b_merge_python_features.R       # Merge Python feature store
│   ├── 19c_rebuild_v3_clean.R            # Clean rebuild — leaker removal, sign fixes
│   ├── 26_tuned_stable_model.R           # Cover model (tuned glmnet, fixed lambda)
│   ├── 36_ensemble_consensus.R           # 3-model cover ensemble
│   ├── 37_kelly_sizing.R                 # Superseded by 45 (kept for the record)
│   ├── 40_nocover_feature_discovery.R    # No-cover feature search
│   ├── 41_nocover_model.R                # No-cover model training
│   ├── 42_nocover_audit.R                # Data integrity audit
│   ├── 43_final_audit.R                  # End-to-end pipeline audit
│   ├── 44_bootstrap_combined.R           # Bootstrap CI + permutation + sensitivity
│   ├── 45_calibrated_kelly.R             # Calibrated probabilities + bet sizing
│   └── run_pipeline.py                   # Python ingest pipeline entry point
├── src/                                  # Python ETL modules (12 sources, 28 tests)
├── data/                                  # Raw + processed datasets
├── configs/                               # Pipeline configs
├── objects/                               # Active model artifacts
│   ├── modeling_data_v3_FINAL.rds        # Final modeling table (3,538 × 300)
│   ├── v3_cover_ensemble.rds             # 3-model cover ensemble preds
│   ├── nocover_ensemble_predictions.rds  # No-cover model preds
│   ├── combined_portfolio_v35.rds        # Final combined bet log
│   ├── cover_preds_v35.rds               # Per-game cover predictions (from 44)
│   ├── nocover_preds_v35.rds             # Per-game no-cover predictions (from 44)
│   ├── calibrators_v35.rds               # Platt calibrators for 2026 sizing
│   └── team_game_full.rds                # Team-game level features
├── results/                               # Active result tables
│   ├── bootstrap_combined_v35.csv        # Final bootstrap validation
│   ├── kelly_calibrated_v35.csv          # Bet sizing comparison
│   ├── kelly_bankroll_paths_v35.csv      # Week-by-week bankroll per sizing
│   ├── kelly_risk_one_season.csv         # One-season risk simulation
│   ├── ensemble_exhaustive_triplets.csv  # Cover ensemble grid search
│   ├── final_exhaustive_with_composites.csv
│   ├── nocover_exhaustive_search.csv
│   └── nocover_feature_correlations.csv
├── memos/                                 # Documentation
│   ├── blog_post.md                      # Full project writeup
│   ├── findings.md                        # Running findings (v3.5 final)
│   ├── feature_summary.md                 # Feature-level analysis
│   ├── data_pipeline_overview.md          # Python pipeline doc
│   ├── no_cover_analysis_findings.md      # No-cover sleeve analysis
│   ├── McCabe_Thomas_memo_1.qmd           # Course memo 1
│   └── McCabe_Thomas_Memo_2.qmd           # Course memo 2
├── figures/                               # Active figures
├── tests/                                 # Python test suite (28 passing)
├── exhibit_beating_the_spread.html        # Interactive HTML exhibit
├── McCabe_Thomas_Final_Report_V2.qmd     # Course deliverable (latest version)
├── pyproject.toml                         # Python project config
└── archive/                               # Earlier development phases (see archive/README.md)
```

## How to Reproduce

```bash
# 0. Pull current-season data, rebuilt team stats, lines/weather, upcoming games
Rscript scripts/02_prepare_recent_data.R

# 1. Ingest raw data (Python)
python scripts/run_pipeline.py

# 2. Build R modeling table
Rscript scripts/01_build_model_data.R
Rscript scripts/19b_merge_python_features.R
Rscript scripts/10_build_advanced_features.R   # NGS, FTN, PFR, injuries, QB (modeling_data_v2)
Rscript scripts/19c_rebuild_v3_clean.R
Rscript scripts/20_finalize_modeling_table.R   # composites (scripts/composites.R) → modeling_data_v36.rds

# 3. Train cover ensemble
Rscript scripts/26_tuned_stable_model.R
Rscript scripts/36_ensemble_consensus.R

# 4. Train no-cover model
Rscript scripts/40_nocover_feature_discovery.R
Rscript scripts/41_nocover_model.R

# 5. Final audit + bootstrap validation
Rscript scripts/43_final_audit.R
Rscript scripts/44_bootstrap_combined.R
Rscript scripts/45_calibrated_kelly.R
# v3.6: MODEL_DATA=modeling_data_v36.rds TAG=v36 for 44 and 45

# 6. This week's picks
Rscript scripts/47_weekly_picks.R
```

## Data Sources (12)

NFL Play-by-Play (nflfastR), Next Gen Stats, FTN charting, PFR Advanced, injury reports, depth charts, FiveThirtyEight Elo, Spreadspoke market & weather, schedules, game scripts, player stats. See `memos/data_pipeline_overview.md`.

## Archive

Earlier scripts (02–35), prior modeling table versions (v2, v3, v4), exploratory result CSVs, and old report drafts are preserved under `archive/` for reproducibility audit. See `archive/README.md`.
