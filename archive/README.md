# Archive

Files from earlier development phases that are not part of the final v3.5 pipeline. Kept for reproducibility and audit but not used by any active script.

## Contents

| Folder | What's in it |
|--------|-------------|
| `scripts/` | Scripts 02–35 (exploratory model search, decay diagnosis, calibration experiments, era analysis, advanced features, XGBoost tests, etc.) |
| `objects/` | Old `.rds` files: prior modeling data versions (`v2`, `v3`, `v4`), individual model strategy snapshots, era datasets, prior ensemble versions |
| `results/` | CSVs from earlier model searches, calibration tests, drift analyses, workflow comparisons |
| `memos/` | `findings_v3.md` and `feature_summary_v3.md` — snapshots taken before v3.5 final |
| `reports/` | Earlier versions of the project report (V1 + Executive Summary). Latest is `McCabe_Thomas_Final_Report_V2.qmd` at the repo root |
| `figures/` | Old tuning grid plots from the initial elastic net / KNN / RF / XGBoost search |

## Why archive instead of delete?

These files document the path to the final result — including the bugs found and fixed along the way. The README + `memos/findings.md` at the repo root reference this history for reviewers who want to verify reproducibility.
