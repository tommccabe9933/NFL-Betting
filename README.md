# Predicting NFL Point Spread Coverage — V3.5 Final

**STAT 301-2: Data Science 2 — Final Project**
**Author:** Thomas McCabe

## Final Result

A combined cover + no-cover ensemble achieves **56.0% against the spread** over **520 out-of-sample bets** across 2023–2025. Bootstrap 90% CI = [52.5%, 59.4%]; entire interval clears the 52.4% breakeven threshold. The 2025 holdout (never seen during development) was profitable: 54.9%, +$860.

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
│   ├── 37_kelly_sizing.R                 # Kelly bankroll simulation
│   ├── 40_nocover_feature_discovery.R    # No-cover feature search
│   ├── 41_nocover_model.R                # No-cover model training
│   ├── 42_nocover_audit.R                # Data integrity audit
│   ├── 43_final_audit.R                  # End-to-end pipeline audit
│   ├── 44_bootstrap_combined.R           # Bootstrap CI + permutation + sensitivity
│   └── run_pipeline.py                   # Python ingest pipeline entry point
├── src/                                  # Python ETL modules (12 sources, 28 tests)
├── data/                                  # Raw + processed datasets
├── configs/                               # Pipeline configs
├── objects/                               # Active model artifacts
│   ├── modeling_data_v3_FINAL.rds        # Final modeling table (3,538 × 300)
│   ├── v3_cover_ensemble.rds             # 3-model cover ensemble preds
│   ├── nocover_ensemble_predictions.rds  # No-cover model preds
│   ├── combined_portfolio_v35.rds        # Final combined bet log
│   └── team_game_full.rds                # Team-game level features
├── results/                               # Active result tables
│   ├── bootstrap_combined_v35.csv        # Final bootstrap validation
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
# 1. Ingest raw data (Python)
python scripts/run_pipeline.py

# 2. Build R modeling table
Rscript scripts/01_build_model_data.R
Rscript scripts/19b_merge_python_features.R
Rscript scripts/19c_rebuild_v3_clean.R

# 3. Train cover ensemble
Rscript scripts/26_tuned_stable_model.R
Rscript scripts/36_ensemble_consensus.R

# 4. Train no-cover model
Rscript scripts/40_nocover_feature_discovery.R
Rscript scripts/41_nocover_model.R

# 5. Final audit + bootstrap validation
Rscript scripts/43_final_audit.R
Rscript scripts/44_bootstrap_combined.R
```

## Data Sources (12)

NFL Play-by-Play (nflfastR), Next Gen Stats, FTN charting, PFR Advanced, injury reports, depth charts, FiveThirtyEight Elo, Spreadspoke market & weather, schedules, game scripts, player stats. See `memos/data_pipeline_overview.md`.

## Archive

Earlier scripts (02–35), prior modeling table versions (v2, v3, v4), exploratory result CSVs, and old report drafts are preserved under `archive/` for reproducibility audit. See `archive/README.md`.
