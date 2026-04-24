# Rebuild all model evaluation using nested temporal walk-forward
# Replaces random 10-fold CV from Stage 1
# Uses log loss as primary tuning metric (rewards calibration)

library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(conflicted)
library(probably)

conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
tidymodels_prefer()

available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

source(here("scripts", "14_research_protocol.R"))

model_data_v2 <- read_rds(here("objects", "modeling_data_v2.rds"))

# ============================================================
# FEATURE SETS (same as before, organized by availability)
# ============================================================

# Universal features (available 2012+)
universal_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  "spread_fav", "over_under_line", "elo_diff",
  "ypp_diff", "pass_eff_diff", "rush_eff_diff", "turnover_diff",
  "fav_roll5_ypp", "dog_roll5_pts", "fav_roll5_pts", "dog_roll5_ypp",
  "pts_play_diff", "third_down_diff", "sack_rate_diff",
  "weather_temperature", "weather_wind_mph",
  "to_momentum_diff", "momentum_diff", "success_rate_diff",
  "off_epa_diff", "def_epa_diff",
  "qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
  "third_down_epa_diff", "redzone_epa_diff",
  "pass_epa_mismatch", "rush_epa_mismatch",
  "explosive_diff", "neg_play_diff",
  "injury_diff", "fav_qb_out", "dog_qb_out",
  "div_game", "is_primetime", "is_dome",
  "rest_diff", "fav_at_home", "fav_short_week", "dog_short_week",
  "stadium_neutral"
)

# Model specs
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("classification")
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |> set_mode("classification")
enet_spec <- logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |> set_mode("classification")

ridge_grid <- grid_regular(penalty(range = c(-5, 1)), levels = 30)
lasso_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)
enet_grid <- grid_regular(
  penalty(range = c(-5, 0)), mixture(range = c(0, 1)),
  levels = c(15, 7)
)

# ============================================================
# RUN WALK-FORWARD EVALUATION (universal features, 2012+ training)
# ============================================================
cat("=== TEMPORAL WALK-FORWARD: Universal features ===\n\n")

cat("Ridge (universal, 2012+):\n")
ridge_uni_results <- walk_forward_evaluate(
  model_data_v2, universal_vars, ridge_spec, ridge_grid,
  min_train_season = 2012, test_seasons = c(2023, 2024, 2025),
  calibrate = TRUE
)

cat("\nLasso (universal, 2012+):\n")
lasso_uni_results <- walk_forward_evaluate(
  model_data_v2, universal_vars, lasso_spec, lasso_grid,
  min_train_season = 2012, test_seasons = c(2023, 2024, 2025),
  calibrate = TRUE
)

cat("\nElastic Net (universal, 2012+):\n")
enet_uni_results <- walk_forward_evaluate(
  model_data_v2, universal_vars, enet_spec, enet_grid,
  min_train_season = 2012, test_seasons = c(2023, 2024, 2025),
  calibrate = TRUE
)

# ============================================================
# ALSO: shorter window (2018+) with universal features
# ============================================================
cat("\n=== TEMPORAL WALK-FORWARD: Universal features, 2018+ ===\n\n")

cat("Ridge (universal, 2018+):\n")
ridge_uni_2018_results <- walk_forward_evaluate(
  model_data_v2, universal_vars, ridge_spec, ridge_grid,
  min_train_season = 2018, test_seasons = c(2023, 2024, 2025),
  calibrate = TRUE
)

# ============================================================
# COMPILE RESULTS
# ============================================================
all_results <- bind_rows(
  ridge_uni_results |> mutate(model = "ridge_universal_2012"),
  lasso_uni_results |> mutate(model = "lasso_universal_2012"),
  enet_uni_results |> mutate(model = "enet_universal_2012"),
  ridge_uni_2018_results |> mutate(model = "ridge_universal_2018")
)

cat("\n\n=== RESULTS: PROPER SCORING RULES ===\n")
cat("(Using calibrated probabilities, tuned on log loss)\n\n")

metrics_by_season <- all_results |>
  group_by(model, test_season) |>
  summarize(
    n = n(),
    brier = round(brier_score(fav_cover, .pred_cover_calibrated, "first"), 4),
    log_loss = round(log_loss(fav_cover, .pred_cover_calibrated, "first"), 4),
    roc_auc = round(tryCatch(roc_auc_vec(fav_cover, .pred_cover_calibrated, "first"),
                             error = function(e) NA), 4),
    raw_accuracy = round(mean((.pred_cover > 0.5) == (fav_cover == "cover")), 3),
    cal_accuracy = round(mean((.pred_cover_calibrated > 0.5) == (fav_cover == "cover")), 3),
    .groups = "drop"
  )

print(metrics_by_season, n = 30)

cat("\n=== OVERALL (pooled across test seasons) ===\n")
metrics_overall <- all_results |>
  group_by(model) |>
  summarize(
    n = n(),
    brier = round(brier_score(fav_cover, .pred_cover_calibrated, "first"), 4),
    log_loss = round(log_loss(fav_cover, .pred_cover_calibrated, "first"), 4),
    roc_auc = round(tryCatch(roc_auc_vec(fav_cover, .pred_cover_calibrated, "first"),
                             error = function(e) NA), 4),
    accuracy = round(mean((.pred_cover_calibrated > 0.5) == (fav_cover == "cover")), 3),
    .groups = "drop"
  ) |>
  arrange(brier)

print(metrics_overall)

cat("\n=== CALIBRATION CHECK ===\n")
cat("If calibrated, predicted probabilities should match observed frequencies\n\n")

for (mod in unique(all_results$model)) {
  cat(mod, ":\n")
  cal <- calibration_curve_data(all_results |> filter(model == mod))
  print(cal)
  cat("\n")
}

# Save
write_rds(all_results, here("objects", "temporal_cv_predictions.rds"))
write_csv(metrics_by_season, here("results", "temporal_cv_season_metrics.csv"))
write_csv(metrics_overall, here("results", "temporal_cv_overall_metrics.csv"))

cat("\nDone! Saved to results/temporal_cv_*.csv\n")
