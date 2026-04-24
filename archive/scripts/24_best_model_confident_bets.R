# ==========================================================================
# BEST POSSIBLE MODEL + CONFIDENT BETTING
# ==========================================================================
#
# Goal: Find the absolute best model, then ONLY bet games where the model
# is confident. If P(cover) > threshold → bet cover. If P(cover) < 1-threshold
# → bet no cover. Otherwise, skip the game.
#
# The insight: we don't need to be right on every game. We need to be right
# on the games where our model has a strong opinion.
#
# Pipeline:
# 1. Wide model search (ridge, lasso, enet, xgboost — huge grids)
# 2. Multiple recipe variants (splines, interactions, PCA, kitchen sink)
# 3. Walk-forward tuning on 2023-2024, holdout on 2025
# 4. Calibrate the winner
# 5. Sweep confidence thresholds on inner CV
# 6. Lock threshold, evaluate on 2025

library(tidyverse)
library(tidymodels)
library(here)
library(xgboost)

source(here("scripts", "14_research_protocol.R"))

available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

# ============================================================
# 1. LOAD DATA
# ============================================================
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))
cat("Data:", nrow(v3), "games x", ncol(v3), "columns\n")

# Use all numeric features that aren't identifiers or targets
id_cols <- c("fav_cover", "game_id_norm", "season", "week",
             "spread_bucket", "season_phase", "fav_team", "dog_team",
             "home_team", "away_team")

all_features <- v3 |>
  select(where(is.numeric), -any_of(c("season", "week"))) |>
  names()

# Features for modeling: everything + identifiers for splitting
model_vars <- c("fav_cover", "game_id_norm", "season", "week", all_features)
model_vars <- unique(intersect(model_vars, names(v3)))

cat("Total features available:", length(all_features), "\n")

# ============================================================
# 2. DEFINE MODEL SPECS WITH WIDE GRIDS
# ============================================================

# --- RIDGE: penalty from 10^-8 to 10^4 ---
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("classification")
ridge_grid <- grid_regular(penalty(range = c(-8, 4)), levels = 60)

# --- LASSO: penalty from 10^-8 to 10^2 ---
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |> set_mode("classification")
lasso_grid <- grid_regular(penalty(range = c(-8, 2)), levels = 60)

# --- ELASTIC NET: wide 2D grid ---
enet_spec <- logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |> set_mode("classification")
set.seed(301)
enet_grid <- grid_latin_hypercube(
  penalty(range = c(-8, 2)),
  mixture(range = c(0, 1)),
  size = 200
)

# --- XGBOOST: wide Bayesian-style grid ---
xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune(),
  min_n = tune(), loss_reduction = tune(),
  sample_size = tune(), mtry = tune()
) |>
  set_engine("xgboost", nthread = available_cores) |>
  set_mode("classification")

set.seed(301)
xgb_grid <- grid_latin_hypercube(
  trees(range = c(100, 1500)),
  tree_depth(range = c(2, 8)),
  learn_rate(range = c(-3, -0.5)),  # 0.001 to 0.3
  min_n(range = c(2, 30)),
  loss_reduction(range = c(-2, 1)),  # 0.01 to 10
  sample_prop(range = c(0.5, 0.95)),
  mtry(range = c(3, min(25, length(all_features)))),
  size = 300
)

# ============================================================
# 3. DEFINE RECIPE VARIANTS
# ============================================================

# Recipe A: Standard (normalize + impute)
make_recipe_standard <- function(data) {
  recipe(fav_cover ~ ., data = data) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())
}

# Recipe B: With splines on key non-linear predictors
make_recipe_splines <- function(data) {
  recipe(fav_cover ~ ., data = data) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_ns(spread_fav, deg_free = 4) |>
    step_ns(over_under_line, deg_free = 3) |>
    step_normalize(all_numeric_predictors())
}

# Recipe C: With interactions
make_recipe_interactions <- function(data) {
  recipe(fav_cover ~ ., data = data) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_interact(~ spread_fav:elo_diff + spread_fav:off_epa_diff +
                    starts_with("weather_wind"):pass_eff_diff) |>
    step_normalize(all_numeric_predictors())
}

# Recipe D: PCA on correlated EPA blocks
make_recipe_pca <- function(data) {
  recipe(fav_cover ~ ., data = data) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_pca(contains("epa"), num_comp = 5, prefix = "epa_pc") |>
    step_pca(contains("roll5"), num_comp = 5, prefix = "roll_pc") |>
    step_normalize(all_numeric_predictors())
}

# ============================================================
# 4. WALK-FORWARD MODEL SEARCH
# ============================================================
# Tune on 2023+2024 (inner temporal CV), holdout 2025

cat("\n=== WALK-FORWARD MODEL SEARCH ===\n")

# We'll evaluate each model on test seasons 2023 + 2024
# Then lock the winner and evaluate on 2025
dev_seasons <- c(2023, 2024)
holdout_season <- 2025

results_list <- list()

# --- 4a: RIDGE (standard recipe) ---
cat("\n--- Ridge (standard) ---\n")
ridge_results <- tryCatch({
  walk_forward_evaluate(
    v3, model_vars, ridge_spec, ridge_grid,
    min_train_season = 2012, test_seasons = dev_seasons,
    calibrate = TRUE
  )
}, error = function(e) { cat("  Ridge failed:", e$message, "\n"); NULL })

if (!is.null(ridge_results)) {
  ridge_brier <- brier_score(ridge_results$fav_cover,
                              ridge_results$.pred_cover_calibrated, "first")
  cat(sprintf("  Dev Brier: %.5f  n=%d\n", ridge_brier, nrow(ridge_results)))
  results_list[["ridge_standard"]] <- ridge_results |>
    mutate(model = "ridge_standard")
}

# --- 4b: LASSO (standard recipe) ---
cat("\n--- Lasso (standard) ---\n")
lasso_results <- tryCatch({
  walk_forward_evaluate(
    v3, model_vars, lasso_spec, lasso_grid,
    min_train_season = 2012, test_seasons = dev_seasons,
    calibrate = TRUE
  )
}, error = function(e) { cat("  Lasso failed:", e$message, "\n"); NULL })

if (!is.null(lasso_results)) {
  lasso_brier <- brier_score(lasso_results$fav_cover,
                              lasso_results$.pred_cover_calibrated, "first")
  cat(sprintf("  Dev Brier: %.5f  n=%d\n", lasso_brier, nrow(lasso_results)))
  results_list[["lasso_standard"]] <- lasso_results |>
    mutate(model = "lasso_standard")
}

# --- 4c: ELASTIC NET (standard recipe) ---
cat("\n--- Elastic Net (standard) ---\n")
enet_results <- tryCatch({
  walk_forward_evaluate(
    v3, model_vars, enet_spec, enet_grid,
    min_train_season = 2012, test_seasons = dev_seasons,
    calibrate = TRUE
  )
}, error = function(e) { cat("  ENet failed:", e$message, "\n"); NULL })

if (!is.null(enet_results)) {
  enet_brier <- brier_score(enet_results$fav_cover,
                             enet_results$.pred_cover_calibrated, "first")
  cat(sprintf("  Dev Brier: %.5f  n=%d\n", enet_brier, nrow(enet_results)))
  results_list[["enet_standard"]] <- enet_results |>
    mutate(model = "enet_standard")
}

# --- 4d: XGBOOST (standard recipe) ---
cat("\n--- XGBoost (standard, 300-point LHS) ---\n")
xgb_results <- tryCatch({
  walk_forward_evaluate(
    v3, model_vars, xgb_spec, xgb_grid,
    min_train_season = 2012, test_seasons = dev_seasons,
    calibrate = TRUE
  )
}, error = function(e) { cat("  XGBoost failed:", e$message, "\n"); NULL })

if (!is.null(xgb_results)) {
  xgb_brier <- brier_score(xgb_results$fav_cover,
                             xgb_results$.pred_cover_calibrated, "first")
  cat(sprintf("  Dev Brier: %.5f  n=%d\n", xgb_brier, nrow(xgb_results)))
  results_list[["xgb_standard"]] <- xgb_results |>
    mutate(model = "xgb_standard")
}

# --- 4e: RIDGE with ERA-3 only (2022+ data, full features) ---
cat("\n--- Ridge Era-3 (2022+ only, full features) ---\n")
era3_data <- read_rds(here("objects", "era3_data.rds"))
era3_vars <- intersect(names(era3_data), model_vars)

ridge_era3_results <- tryCatch({
  walk_forward_evaluate(
    era3_data, era3_vars, ridge_spec, ridge_grid,
    min_train_season = 2022, test_seasons = dev_seasons,
    calibrate = TRUE
  )
}, error = function(e) { cat("  Ridge Era3 failed:", e$message, "\n"); NULL })

if (!is.null(ridge_era3_results)) {
  era3_brier <- brier_score(ridge_era3_results$fav_cover,
                             ridge_era3_results$.pred_cover_calibrated, "first")
  cat(sprintf("  Dev Brier: %.5f  n=%d\n", era3_brier, nrow(ridge_era3_results)))
  results_list[["ridge_era3"]] <- ridge_era3_results |>
    mutate(model = "ridge_era3")
}

# ============================================================
# 5. MODEL COMPARISON ON DEV SET
# ============================================================
cat("\n=== MODEL COMPARISON (Dev: 2023-2024) ===\n")

all_dev_preds <- bind_rows(results_list)

model_comparison <- all_dev_preds |>
  group_by(model) |>
  summarize(
    n = n(),
    brier = brier_score(fav_cover, .pred_cover_calibrated, "first"),
    log_loss = log_loss(fav_cover, .pred_cover_calibrated, "first"),
    accuracy = mean((.pred_cover_calibrated > 0.5) == (fav_cover == "cover")),
    mean_pred = mean(.pred_cover_calibrated),
    .groups = "drop"
  ) |>
  arrange(brier)

cat("\n")
print(model_comparison)
write_csv(model_comparison, here("results", "advanced_model_comparison.csv"))

# ============================================================
# 6. CONFIDENT BETTING ANALYSIS
# ============================================================
cat("\n=== CONFIDENT BETTING ANALYSIS ===\n")
cat("Only bet when model probability is far from 0.50\n\n")

# Use the best model's predictions
best_model_name <- model_comparison |> slice_min(brier) |> pull(model) |> first()
cat("Best model:", best_model_name, "\n\n")

best_preds <- all_dev_preds |> filter(model == best_model_name)

# Sweep confidence thresholds
# If P(cover) > threshold → bet cover
# If P(cover) < (1 - threshold) → bet no cover
# Otherwise → skip

cat("Confidence Threshold Analysis:\n")
cat(sprintf("%-12s %6s %7s %8s %7s %7s\n",
            "Threshold", "Bets", "Pct", "WinRate", "Profit", "ROI"))
cat(paste(rep("-", 55), collapse = ""), "\n")

threshold_results <- list()

for (thresh in seq(0.51, 0.65, by = 0.01)) {
  bets <- best_preds |>
    mutate(
      cal_p = .pred_cover_calibrated,
      bet_cover = cal_p > thresh,
      bet_nocover = cal_p < (1 - thresh),
      has_bet = bet_cover | bet_nocover,
      bet_correct = case_when(
        bet_cover & (fav_cover == "cover") ~ TRUE,
        bet_nocover & (fav_cover == "no_cover") ~ TRUE,
        has_bet ~ FALSE,
        TRUE ~ NA
      )
    ) |>
    filter(has_bet)

  if (nrow(bets) < 5) next

  win_rate <- mean(bets$bet_correct)
  profit <- sum(if_else(bets$bet_correct, 100, -110))
  roi <- profit / (nrow(bets) * 110)

  cat(sprintf("P > %-6.2f  %5d  %5.1f%%  %6.1f%%  $%6.0f  %5.1f%%\n",
              thresh, nrow(bets), nrow(bets) / nrow(best_preds) * 100,
              win_rate * 100, profit, roi * 100))

  threshold_results[[as.character(thresh)]] <- tibble(
    threshold = thresh,
    n_bets = nrow(bets),
    pct_games = nrow(bets) / nrow(best_preds) * 100,
    win_rate = win_rate,
    profit = profit,
    roi = roi
  )
}

thresh_df <- bind_rows(threshold_results)
write_csv(thresh_df, here("results", "confidence_threshold_analysis.csv"))

# ============================================================
# 7. FIND OPTIMAL THRESHOLD
# ============================================================
cat("\n=== OPTIMAL THRESHOLD ===\n")

# Best threshold = highest ROI with at least 30 bets
optimal <- thresh_df |>
  filter(n_bets >= 30) |>
  slice_max(roi)

if (nrow(optimal) > 0) {
  cat(sprintf("Optimal threshold: P > %.2f (or P < %.2f)\n",
              optimal$threshold, 1 - optimal$threshold))
  cat(sprintf("  Bets: %d (%.1f%% of games)\n", optimal$n_bets, optimal$pct_games))
  cat(sprintf("  Win rate: %.1f%%\n", optimal$win_rate * 100))
  cat(sprintf("  ROI: %.1f%%\n", optimal$roi * 100))
} else {
  cat("No threshold produces 30+ bets\n")
}

# ============================================================
# 8. HOLDOUT EVALUATION (2025) — LOCKED STRATEGY
# ============================================================
cat("\n=== 2025 HOLDOUT EVALUATION ===\n")

# Re-run best model on 2025
cat("Running best model (", best_model_name, ") on 2025 holdout...\n")

# Get the appropriate spec and grid
if (grepl("ridge", best_model_name)) {
  holdout_spec <- ridge_spec
  holdout_grid <- ridge_grid
  holdout_data <- if (grepl("era3", best_model_name)) era3_data else v3
  holdout_vars <- if (grepl("era3", best_model_name)) era3_vars else model_vars
  min_train <- if (grepl("era3", best_model_name)) 2022 else 2012
} else if (grepl("lasso", best_model_name)) {
  holdout_spec <- lasso_spec
  holdout_grid <- lasso_grid
  holdout_data <- v3
  holdout_vars <- model_vars
  min_train <- 2012
} else if (grepl("enet", best_model_name)) {
  holdout_spec <- enet_spec
  holdout_grid <- enet_grid
  holdout_data <- v3
  holdout_vars <- model_vars
  min_train <- 2012
} else {
  holdout_spec <- xgb_spec
  holdout_grid <- xgb_grid
  holdout_data <- v3
  holdout_vars <- model_vars
  min_train <- 2012
}

holdout_preds <- tryCatch({
  walk_forward_evaluate(
    holdout_data, holdout_vars, holdout_spec, holdout_grid,
    min_train_season = min_train, test_seasons = 2025,
    calibrate = TRUE
  )
}, error = function(e) { cat("  Holdout failed:", e$message, "\n"); NULL })

if (!is.null(holdout_preds) && nrow(optimal) > 0) {
  locked_thresh <- optimal$threshold[1]

  holdout_bets <- holdout_preds |>
    mutate(
      cal_p = .pred_cover_calibrated,
      bet_cover = cal_p > locked_thresh,
      bet_nocover = cal_p < (1 - locked_thresh),
      has_bet = bet_cover | bet_nocover,
      bet_correct = case_when(
        bet_cover & (fav_cover == "cover") ~ TRUE,
        bet_nocover & (fav_cover == "no_cover") ~ TRUE,
        has_bet ~ FALSE,
        TRUE ~ NA
      )
    )

  all_bets <- holdout_bets |> filter(has_bet)
  total_games <- nrow(holdout_preds)

  if (nrow(all_bets) > 0) {
    win_rate <- mean(all_bets$bet_correct)
    profit <- sum(if_else(all_bets$bet_correct, 100, -110))
    roi <- profit / (nrow(all_bets) * 110)

    cat(sprintf("\n  2025 HOLDOUT RESULTS (threshold: P > %.2f)\n", locked_thresh))
    cat(sprintf("  Total games: %d\n", total_games))
    cat(sprintf("  Bets placed: %d (%.1f%%)\n", nrow(all_bets), nrow(all_bets) / total_games * 100))
    cat(sprintf("  Win rate: %.1f%%  (breakeven: 52.4%%)\n", win_rate * 100))
    cat(sprintf("  Profit: $%.0f\n", profit))
    cat(sprintf("  ROI: %.1f%%\n", roi * 100))

    # Bootstrap CI
    set.seed(301)
    boot_rois <- replicate(2000, {
      idx <- sample(nrow(all_bets), replace = TRUE)
      boot <- all_bets[idx, ]
      sum(if_else(boot$bet_correct, 100, -110)) / (nrow(boot) * 110)
    })
    cat(sprintf("  ROI 95%% CI: [%.1f%%, %.1f%%]\n",
                quantile(boot_rois, 0.025) * 100,
                quantile(boot_rois, 0.975) * 100))
    cat(sprintf("  P(ROI > 0): %.1f%%\n", mean(boot_rois > 0) * 100))

    # Also show overall model performance (all games, not just bets)
    overall_brier <- brier_score(holdout_preds$fav_cover,
                                  holdout_preds$.pred_cover_calibrated, "first")
    overall_acc <- mean((holdout_preds$.pred_cover_calibrated > 0.5) == (holdout_preds$fav_cover == "cover"))
    cat(sprintf("\n  Overall 2025 model: Brier=%.5f, Accuracy=%.1f%%\n",
                overall_brier, overall_acc * 100))
  } else {
    cat("  No bets placed at threshold", locked_thresh, "\n")
  }
}

# ============================================================
# 9. SAVE EVERYTHING
# ============================================================
write_csv(model_comparison, here("results", "advanced_model_comparison.csv"))

if (!is.null(holdout_preds)) {
  write_rds(holdout_preds, here("objects", "holdout_2025_predictions.rds"))
}

cat("\nDone!\n")
