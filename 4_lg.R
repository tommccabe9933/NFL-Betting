library(tidymodels)

log_mod <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

log_workflow <- workflow() |>
  add_model(log_mod) |>
  add_recipe(basic_recipe)

log_results <- fit_resamples(
  log_workflow,
  resamples = cv_folds,
  metrics = metric_set(roc_auc, accuracy)
)