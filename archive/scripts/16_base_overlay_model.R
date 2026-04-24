# Base + Overlay Architecture
# Stage A: Base model on universal features (2012+)
# Stage B: Overlay model on FTN/NGS/PFR residuals (2022+ only)
# Combined: inverse_logit(base_logit + overlay_adjustment)

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

source(here("scripts", "14_research_protocol.R"))

model_data_v2 <- read_rds(here("objects", "modeling_data_v2.rds"))

# ============================================================
# FEATURE GROUPS (natural blocks for grouped shrinkage)
# ============================================================

# Group 1: Market (always available)
market_vars <- c("spread_fav", "over_under_line", "elo_diff")

# Group 2: Team efficiency (always available)
efficiency_vars <- c("ypp_diff", "pass_eff_diff", "rush_eff_diff", "turnover_diff",
                     "fav_roll5_ypp", "dog_roll5_pts", "fav_roll5_pts", "dog_roll5_ypp",
                     "pts_play_diff", "third_down_diff", "sack_rate_diff",
                     "to_momentum_diff", "momentum_diff", "success_rate_diff")

# Group 3: QB quality (available 2012+)
qb_vars <- c("qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff")

# Group 4: Situational EPA (available 2012+)
situational_vars <- c("third_down_epa_diff", "redzone_epa_diff",
                       "pass_epa_mismatch", "rush_epa_mismatch",
                       "explosive_diff", "neg_play_diff",
                       "off_epa_diff", "def_epa_diff")

# Group 5: Injuries (available 2012+)
injury_vars <- c("injury_diff", "fav_qb_out", "dog_qb_out")

# Group 6: Schedule context (available 2012+)
context_vars <- c("div_game", "is_primetime", "is_dome",
                   "rest_diff", "fav_at_home", "fav_short_week", "dog_short_week",
                   "stadium_neutral", "weather_temperature", "weather_wind_mph")

# BASE features: all universal (Groups 1-6)
base_vars <- c("fav_cover", "game_id_norm", "season", "week",
               market_vars, efficiency_vars, qb_vars, situational_vars,
               injury_vars, context_vars)

# OVERLAY features: FTN + NGS + PFR (Groups 7-9, 2022+ only)
overlay_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  # FTN (2022+)
  "fav_ftn_roll5_blitz", "fav_ftn_roll5_rpo",
  "fav_ftn_roll5_play_action", "fav_ftn_roll5_pass_rush",
  "dog_ftn_roll5_blitz", "dog_ftn_roll5_rpo",
  "dog_ftn_roll5_play_action", "dog_ftn_roll5_pass_rush",
  # NGS (2021+)
  "fav_ngs_roll5_cpoe", "fav_ngs_roll5_rush_yoe", "fav_ngs_roll5_rush_eff",
  "fav_ngs_roll5_ttt", "fav_ngs_roll5_aggressiveness",
  "dog_ngs_roll5_cpoe", "dog_ngs_roll5_rush_yoe", "dog_ngs_roll5_rush_eff",
  "dog_ngs_roll5_ttt", "dog_ngs_roll5_aggressiveness",
  # PFR (2018+)
  "fav_pfr_roll5_pressure_rate", "fav_pfr_roll5_drop_rate", "fav_pfr_roll5_bad_throw_rate",
  "dog_pfr_roll5_pressure_rate", "dog_pfr_roll5_drop_rate", "dog_pfr_roll5_bad_throw_rate",
  "fav_pfr_roll5_def_pressure", "dog_pfr_roll5_def_pressure",
  "fav_pfr_roll5_def_comp_allowed", "dog_pfr_roll5_def_comp_allowed"
)

# Model specs
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("classification")
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |> set_mode("classification")

ridge_grid <- grid_regular(penalty(range = c(-5, 1)), levels = 30)
lasso_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)

# ============================================================
# WALK-FORWARD: BASE MODEL (universal features, 2012+)
# ============================================================
cat("=== STAGE A: BASE MODEL (universal, 2012+) ===\n\n")

base_results <- list()

for (test_s in c(2023, 2024, 2025)) {
  cat("Test season", test_s, ":\n")

  train <- model_data_v2 |> filter(season < test_s) |> select(any_of(base_vars))
  test <- model_data_v2 |> filter(season == test_s) |> select(any_of(base_vars))

  # Inner temporal CV for tuning
  inner_cv <- make_temporal_cv(train, min_train_season = 2012)

  recipe_base <- recipe(fav_cover ~ ., data = train) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

  wf <- workflow() |> add_model(ridge_spec) |> add_recipe(recipe_base)

  set.seed(301)
  tune_result <- tune_grid(wf, resamples = inner_cv, grid = ridge_grid,
                           metrics = metric_set(roc_auc, mn_log_loss),
                           control = control_grid(save_pred = TRUE, verbose = FALSE))

  best <- select_best(tune_result, metric = "mn_log_loss")

  # Calibrate on inner validation predictions
  inner_preds <- collect_predictions(tune_result) |>
    semi_join(best, by = intersect(names(best), names(collect_predictions(tune_result))))
  cal_fit <- fit_platt(inner_preds)

  # Fit on full training data
  final_wf <- finalize_workflow(wf, best)
  final_fit <- fit(final_wf, data = train)

  # Predict test
  test_preds <- test |>
    select(game_id_norm, season, week, fav_cover) |>
    bind_cols(predict(final_fit, new_data = test, type = "prob"))

  # Calibrate
  test_preds <- apply_platt(test_preds, cal_fit)

  # Compute base logits for overlay input
  test_preds <- test_preds |>
    mutate(base_logit = log(.pred_cover_calibrated / (1 - .pred_cover_calibrated)))

  base_results[[as.character(test_s)]] <- test_preds

  brier <- brier_score(test_preds$fav_cover, test_preds$.pred_cover_calibrated, "first")
  acc <- mean((test_preds$.pred_cover_calibrated > 0.5) == (test_preds$fav_cover == "cover"))
  cat("  Brier:", round(brier, 4), "| Accuracy:", round(acc, 3), "\n")
}

base_all <- bind_rows(base_results)

# ============================================================
# WALK-FORWARD: OVERLAY MODEL (FTN/NGS/PFR, train 2022+ only)
# Predicts RESIDUAL of base model
# ============================================================
cat("\n=== STAGE B: OVERLAY MODEL (FTN/NGS/PFR, 2022+) ===\n")
cat("Target: where was the base model wrong?\n\n")

overlay_results <- list()

for (test_s in c(2024, 2025)) {
  cat("Test season", test_s, ":\n")

  # Get base model predictions for training window (2022 to test_s-1)
  # We need to run base model on 2022-2023 to get residuals
  base_train_preds <- list()
  for (res_s in 2022:(test_s - 1)) {
    # Quick base model prediction for residual season
    res_train <- model_data_v2 |> filter(season < res_s) |> select(any_of(base_vars))
    res_test <- model_data_v2 |> filter(season == res_s) |> select(any_of(base_vars))

    recipe_res <- recipe(fav_cover ~ ., data = res_train) |>
      update_role(game_id_norm, season, week, new_role = "id variable") |>
      step_zv(all_predictors()) |>
      step_impute_mean(all_numeric_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_normalize(all_numeric_predictors())

    # Use a reasonable fixed penalty (from base model tuning)
    wf_res <- workflow() |>
      add_model(logistic_reg(penalty = 0.01, mixture = 0) |>
                  set_engine("glmnet") |> set_mode("classification")) |>
      add_recipe(recipe_res)
    fit_res <- fit(wf_res, data = res_train)

    res_preds <- res_test |>
      select(game_id_norm, season, week, fav_cover) |>
      bind_cols(predict(fit_res, new_data = res_test, type = "prob")) |>
      mutate(base_logit = log(.pred_cover / (1 - .pred_cover)),
             # Did the base model get it wrong? (residual target)
             outcome_num = as.integer(fav_cover == "cover"),
             base_residual = outcome_num - .pred_cover)

    base_train_preds[[as.character(res_s)]] <- res_preds
  }

  base_train_all <- bind_rows(base_train_preds)

  # Build overlay training data: overlay features + base residual
  overlay_train <- model_data_v2 |>
    filter(season %in% 2022:(test_s - 1)) |>
    select(any_of(overlay_vars)) |>
    left_join(base_train_all |> select(game_id_norm, base_logit, base_residual),
              by = "game_id_norm") |>
    filter(!is.na(base_logit))

  # Overlay test data
  overlay_test <- model_data_v2 |>
    filter(season == test_s) |>
    select(any_of(overlay_vars))

  base_test_logits <- base_results[[as.character(test_s)]] |>
    select(game_id_norm, base_logit, .pred_cover_calibrated)

  overlay_test <- overlay_test |>
    left_join(base_test_logits, by = "game_id_norm")

  if (nrow(overlay_train) < 100) {
    cat("  Skipping - only", nrow(overlay_train), "overlay training rows\n")
    next
  }

  # Train overlay: predict fav_cover using overlay features + base_logit
  # The base_logit acts as an offset — overlay learns INCREMENTAL signal
  recipe_overlay <- recipe(fav_cover ~ ., data = overlay_train |> select(-base_residual)) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

  wf_overlay <- workflow() |> add_model(lasso_spec) |> add_recipe(recipe_overlay)

  # Simple temporal CV (small data)
  inner_cv_overlay <- make_temporal_cv(overlay_train, min_train_season = 2022)

  set.seed(301)
  tune_overlay <- tune_grid(wf_overlay, resamples = inner_cv_overlay, grid = lasso_grid,
                            metrics = metric_set(roc_auc, mn_log_loss),
                            control = control_grid(save_pred = FALSE, verbose = FALSE))

  best_overlay <- select_best(tune_overlay, metric = "mn_log_loss")
  final_overlay <- finalize_workflow(wf_overlay, best_overlay)
  fit_overlay <- fit(final_overlay, data = overlay_train |> select(-base_residual))

  # Predict
  overlay_preds <- overlay_test |>
    select(game_id_norm, season, week, fav_cover) |>
    bind_cols(predict(fit_overlay, new_data = overlay_test, type = "prob")) |>
    rename(.pred_cover_combined = .pred_cover) |>
    left_join(base_test_logits |> select(game_id_norm, .pred_cover_calibrated),
              by = "game_id_norm") |>
    rename(base_prob = .pred_cover_calibrated)

  overlay_results[[as.character(test_s)]] <- overlay_preds

  # Compare base vs combined
  base_acc <- mean((overlay_preds$base_prob > 0.5) == (overlay_preds$fav_cover == "cover"))
  combined_acc <- mean((overlay_preds$.pred_cover_combined > 0.5) == (overlay_preds$fav_cover == "cover"))
  base_brier <- brier_score(overlay_preds$fav_cover, overlay_preds$base_prob, "first")
  combined_brier <- brier_score(overlay_preds$fav_cover, overlay_preds$.pred_cover_combined, "first")

  cat("  Base:     Brier:", round(base_brier, 4), "| Acc:", round(base_acc, 3), "\n")
  cat("  Combined: Brier:", round(combined_brier, 4), "| Acc:", round(combined_acc, 3), "\n")
  cat("  Lift:     Brier:", round(base_brier - combined_brier, 4),
      "| Acc:", round(combined_acc - base_acc, 3), "\n")
}

# ============================================================
# FINAL COMPARISON
# ============================================================
cat("\n=== FINAL COMPARISON (2024-2025) ===\n")

if (length(overlay_results) > 0) {
  overlay_all <- bind_rows(overlay_results)

  cat("\nBase model (calibrated):\n")
  base_2425 <- base_all |> filter(season %in% c(2024, 2025))
  cat("  Brier:", round(brier_score(base_2425$fav_cover, base_2425$.pred_cover_calibrated, "first"), 4), "\n")
  cat("  Accuracy:", round(mean((base_2425$.pred_cover_calibrated > 0.5) == (base_2425$fav_cover == "cover")), 3), "\n")

  cat("\nBase + Overlay:\n")
  cat("  Brier:", round(brier_score(overlay_all$fav_cover, overlay_all$.pred_cover_combined, "first"), 4), "\n")
  cat("  Accuracy:", round(mean((overlay_all$.pred_cover_combined > 0.5) == (overlay_all$fav_cover == "cover")), 3), "\n")
}

# Save
write_rds(base_all, here("objects", "base_model_predictions.rds"))
if (length(overlay_results) > 0) {
  write_rds(overlay_all, here("objects", "overlay_model_predictions.rds"))
}

cat("\nDone!\n")
