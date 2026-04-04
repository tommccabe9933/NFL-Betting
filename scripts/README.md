# Scripts

My Rscripts that form the modeling pipeline.

| Script | Purpose |
|--------|---------|
| `00_project_setup.R` | Shared setup: libraries, helper functions, file paths |
| `01_build_model_data.R` | Load raw data, clean, engineer features, save `modeling_data.rds` |
| `02_prepare_modeling_objects.R` | Train/test split, CV folds, recipe definitions |
| `03_define_workflows.R` | Define all model workflows (calls 03a-03d) |
| `04_fit_workflows.R` | Fit all workflows via cross-validation (calls 04a-04d) |
| `05_collect_results.R` | Collect and compare CV results across workflows |
| `06_fit_expanded_workflows.R` | Fit expanded/curated recipe workflows |
| `07_audit_and_select_final_model.R` | Select best model, evaluate on held-out test set |
| `08_make_report_assets.R` | Generate figures and tables for the report |
