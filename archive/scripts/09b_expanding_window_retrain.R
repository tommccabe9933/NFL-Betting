# Expanding window retraining: retrain ridge_curated each season

library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(conflicted)

conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
tidymodels_prefer()

available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

model_data <- read_rds(here("objects", "modeling_data.rds"))

# Same curated variables as the original pipeline (02_prepare_modeling_objects.R:83-88)
curated_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  "fav_roll5_ypp", "ypp_diff", "turnover_diff", "dog_roll5_pts",
  "weather_temperature", "off_epa_diff", "dog_short_week",
  "elo_diff", "def_epa_diff", "to_momentum_diff", "success_rate_diff"
)

# Same ridge spec (03b_define_regularized_models.R:7-9)
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("classification")

# Same tuning grid (03b:168-170)
ridge_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  levels = 25
)

# Seasons to predict
test_seasons <- sort(unique(model_data$season[model_data$season >= 2021]))

retrain_for_season <- function(target_season, model_data, curated_vars, ridge_spec, ridge_grid) {
  cat("=== Retraining for season", target_season, "===\n")

  # Expanding window: train on all prior seasons
  train <- model_data |> filter(season < target_season) |> select(any_of(curated_vars))
  test <- model_data |> filter(season == target_season) |> select(any_of(curated_vars))

  cat("  Train:", nrow(train), "rows (seasons",
      min(train$season, na.rm = TRUE), "-", max(train$season, na.rm = TRUE), ")\n")
  cat("  Test:", nrow(test), "rows (season", target_season, ")\n")

  # Fresh recipe with normalization from THIS training window
  recipe_fresh <- recipe(fav_cover ~ ., data = train) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

  wf <- workflow() |>
    add_model(ridge_spec) |>
    add_recipe(recipe_fresh)

  # 10-fold CV, 3 repeats, stratified
  set.seed(301)
  cv_folds <- vfold_cv(train, v = 10, repeats = 3, strata = fav_cover)

  # Tune
  cat("  Tuning (750 model fits)...\n")
  tune_result <- tune_grid(
    wf,
    resamples = cv_folds,
    grid = ridge_grid,
    metrics = metric_set(roc_auc, accuracy),
    control = control_grid(save_pred = FALSE, verbose = FALSE)
  )

  best_params <- select_best(tune_result, metric = "roc_auc")
  best_cv_metrics <- collect_metrics(tune_result) |>
    semi_join(best_params, by = "penalty") |>
    filter(.metric == "roc_auc")

  cat("  Best penalty:", round(best_params$penalty, 5),
      "| CV ROC AUC:", round(best_cv_metrics$mean, 4), "\n")

  # Finalize and fit on full training window
  final_wf <- finalize_workflow(wf, best_params)
  final_fit <- fit(final_wf, data = train)

  # Predict on target season
  pred_probs <- predict(final_fit, new_data = test, type = "prob")
  pred_class <- predict(final_fit, new_data = test, type = "class")

  predictions <- test |>
    select(game_id_norm, season, week, fav_cover) |>
    bind_cols(pred_probs, pred_class) |>
    rename(.pred_cover = .pred_cover, .pred_class = .pred_class) |>
    mutate(
      train_seasons = paste0("2012-", target_season - 1),
      train_n = nrow(train),
      best_penalty = best_params$penalty,
      cv_roc_auc = best_cv_metrics$mean
    )

  cat("  Done.\n\n")
  predictions
}

# Run expanding window retraining for each test season
cat("Starting expanding window retraining...\n\n")

all_predictions <- map(
  test_seasons,
  ~ retrain_for_season(.x, model_data, curated_vars, ridge_spec, ridge_grid)
) |>
  bind_rows()

cat("Total predictions:", nrow(all_predictions), "\n")

# Save predictions
write_rds(all_predictions, here("objects", "expanding_window_predictions.rds"))

# Save tuning log
tuning_log <- all_predictions |>
  distinct(season, train_seasons, train_n, best_penalty, cv_roc_auc)

write_csv(tuning_log, here("results", "expanding_tuning_log.csv"))

cat("\nTuning log:\n")
print(tuning_log)

cat("\nDone. Outputs: objects/expanding_window_predictions.rds, results/expanding_tuning_log.csv\n")
