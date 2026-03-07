library(tidymodels)

null_mod <- null_model() |>
  set_engine("parsnip") |>
  set_mode("classification")

null_workflow <- workflow() |>
  add_model(null_mod) |>
  add_recipe(basic_recipe)

null_results <- fit_resamples(
  null_workflow,
  resamples = cv_folds,
  metrics = metric_set(roc_auc, accuracy)
)