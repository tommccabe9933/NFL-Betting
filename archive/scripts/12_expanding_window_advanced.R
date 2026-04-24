# Expanding window retraining WITH advanced features
# Key insight: FTN/NGS features only exist from 2021+, so train windows
# that include 2022+ can actually USE these features.

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

model_data_v2 <- read_rds(here("objects", "modeling_data_v2.rds"))

# Advanced curated variables (includes new features)
advanced_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  # Original strong
  "spread_fav", "elo_diff", "ypp_diff", "turnover_diff",
  "fav_roll5_ypp", "dog_roll5_pts", "weather_temperature",
  "to_momentum_diff", "success_rate_diff",
  # QB quality
  "qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
  # Situational EPA
  "third_down_epa_diff", "redzone_epa_diff",
  "pass_epa_mismatch", "rush_epa_mismatch",
  "explosive_diff", "neg_play_diff",
  # Injury
  "injury_diff", "fav_qb_out", "dog_qb_out",
  # Context
  "div_game", "is_primetime", "is_dome",
  "dog_short_week",
  # EPA diffs (fixed)
  "off_epa_diff", "def_epa_diff"
)

# Model specs
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("classification")

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |> set_mode("classification")

ridge_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 25)
lasso_grid <- grid_regular(penalty(range = c(-4, -0.5)), levels = 25)

test_seasons <- sort(unique(model_data_v2$season[model_data_v2$season >= 2021]))

retrain_season <- function(target_season, data, vars, spec, grid, model_label) {
  cat("  [", model_label, "] Season", target_season, "...")

  train <- data |> filter(season < target_season) |> select(any_of(vars))
  test <- data |> filter(season == target_season) |> select(any_of(vars))

  recipe_fresh <- recipe(fav_cover ~ ., data = train) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

  wf <- workflow() |> add_model(spec) |> add_recipe(recipe_fresh)

  set.seed(301)
  cv_folds <- vfold_cv(train, v = 10, repeats = 3, strata = fav_cover)

  tune_result <- tune_grid(wf, resamples = cv_folds, grid = grid,
                           metrics = metric_set(roc_auc, accuracy),
                           control = control_grid(save_pred = FALSE, verbose = FALSE))

  best <- select_best(tune_result, metric = "roc_auc")
  cv_auc <- (collect_metrics(tune_result) |>
    semi_join(best, by = "penalty") |> filter(.metric == "roc_auc"))$mean

  final_wf <- finalize_workflow(wf, best)
  final_fit <- fit(final_wf, data = train)

  pred_probs <- predict(final_fit, new_data = test, type = "prob")
  pred_class <- predict(final_fit, new_data = test, type = "class")

  preds <- test |>
    select(game_id_norm, season, week, fav_cover) |>
    bind_cols(pred_probs, pred_class) |>
    mutate(model = model_label, train_seasons = paste0("2012-", target_season - 1),
           best_penalty = best$penalty, cv_roc_auc = cv_auc)

  cat(" CV AUC:", round(cv_auc, 4), "| penalty:", round(best$penalty, 5), "\n")
  preds
}

cat("=== EXPANDING WINDOW WITH ADVANCED FEATURES ===\n\n")

# Run both ridge and lasso with advanced features
all_preds <- list()

for (yr in test_seasons) {
  cat("Season", yr, ":\n")
  all_preds[[paste0("ridge_adv_", yr)]] <- retrain_season(
    yr, model_data_v2, advanced_vars, ridge_spec, ridge_grid, "ridge_advanced")
  all_preds[[paste0("lasso_adv_", yr)]] <- retrain_season(
    yr, model_data_v2, advanced_vars, lasso_spec, lasso_grid, "lasso_advanced")
}

all_preds_df <- bind_rows(all_preds)

# Also load the original expanding window predictions for comparison
orig_expanding <- read_rds(here("objects", "expanding_window_predictions.rds")) |>
  mutate(model = "ridge_original")

# Combine
combined <- bind_rows(
  all_preds_df |> select(game_id_norm, season, week, fav_cover, .pred_cover, .pred_class, model),
  orig_expanding |> select(game_id_norm, season, week, fav_cover, .pred_cover, .pred_class, model)
)

# Compute metrics
cat("\n=== RESULTS BY MODEL AND SEASON ===\n")
results <- combined |>
  mutate(
    correct = as.integer((.pred_cover > 0.5) == (fav_cover == "cover")),
    profit = if_else(correct == 1L, 100L, -110L)
  ) |>
  group_by(model, season) |>
  summarize(
    n = n(),
    accuracy = round(mean(correct), 3),
    roc_auc = round(roc_auc_vec(fav_cover, .pred_cover, event_level = "first"), 4),
    profit = sum(profit),
    roi_pct = round(sum(profit) / (n() * 110) * 100, 1),
    .groups = "drop"
  )

print(results, n = 30)

cat("\n=== OVERALL BY MODEL ===\n")
overall <- combined |>
  mutate(
    correct = as.integer((.pred_cover > 0.5) == (fav_cover == "cover")),
    profit = if_else(correct == 1L, 100L, -110L)
  ) |>
  group_by(model) |>
  summarize(
    n = n(),
    accuracy = round(mean(correct), 3),
    roc_auc = round(roc_auc_vec(fav_cover, .pred_cover, event_level = "first"), 4),
    total_profit = sum(profit),
    roi_pct = round(sum(profit) / (n() * 110) * 100, 1),
    .groups = "drop"
  ) |>
  arrange(desc(roc_auc))

print(overall)

# Save
write_csv(results, here("results", "expanding_advanced_comparison.csv"))
write_csv(overall, here("results", "expanding_advanced_overall.csv"))
write_rds(all_preds_df, here("objects", "expanding_advanced_predictions.rds"))

cat("\nDone!\n")
