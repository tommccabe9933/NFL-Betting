# Objects

Saved RDS files produced by the modeling pipeline. These contain fitted model objects, prediction outputs, and intermediate data used by the report.

Key files:

- `modeling_data.rds` - Final modeling dataset (3,538 games)
- `prepared_modeling_objects.rds` - Train/test data, CV folds, recipe objects
- `workflow_results.rds` - Cross-validation results for all tuned workflows
- `best_workflow_test_predictions.rds` - Test set predictions from the selected model

This directory is git-ignored due to file size. Re-run the scripts in `scripts/` to regenerate.
