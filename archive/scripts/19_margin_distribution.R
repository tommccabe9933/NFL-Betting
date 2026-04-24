# Margin Distribution Model
# Predicts actual_margin - spread as continuous variable
# Uses Huber loss (robust to fat tails) and quantile regression
# Outputs predictive distribution, not just point estimate

library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(conflicted)
library(quantreg)

conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
tidymodels_prefer()

available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

source(here("scripts", "14_research_protocol.R"))

model_data_v2 <- read_rds(here("objects", "modeling_data_v2.rds"))
team_game_full <- read_rds(here("objects", "team_game_full.rds"))

# ============================================================
# BUILD MARGIN TARGET: actual_margin - spread
# ============================================================
cat("Building margin-vs-spread target...\n")

# Get actual game margins from team_game_full (favorite perspective)
margins <- team_game_full |>
  filter(team_norm == team_favorite_id, cover != "push") |>
  transmute(
    game_id_norm,
    fav_score = team_score,
    dog_score = opponent_score,
    actual_margin = team_score - opponent_score,
    spread = spread_favorite,
    margin_vs_spread = actual_margin + spread_favorite  # positive = cover
  )

# Merge into model_data_v2
margin_data <- model_data_v2 |>
  left_join(margins |> select(game_id_norm, actual_margin, margin_vs_spread),
            by = "game_id_norm") |>
  filter(!is.na(margin_vs_spread))

cat("Margin data:", nrow(margin_data), "games\n")
cat("Margin vs spread: mean =", round(mean(margin_data$margin_vs_spread), 2),
    "sd =", round(sd(margin_data$margin_vs_spread), 2), "\n")
cat("Distribution of margin_vs_spread:\n")
print(quantile(margin_data$margin_vs_spread, probs = c(0.05, 0.25, 0.5, 0.75, 0.95)))

# Key numbers analysis
cat("\nNFL key number proximity:\n")
key_nums <- c(-14, -10, -7, -3, 0, 3, 7, 10, 14)
for (kn in key_nums) {
  pct <- mean(abs(margin_data$margin_vs_spread - kn) <= 0.5) * 100
  cat(sprintf("  Within 0.5 of %+d: %.1f%%\n", kn, pct))
}

# Universal features for regression
reg_vars <- c(
  "margin_vs_spread", "game_id_norm", "season", "week",
  "spread_fav", "over_under_line", "elo_diff",
  "ypp_diff", "pass_eff_diff", "rush_eff_diff", "turnover_diff",
  "fav_roll5_ypp", "dog_roll5_pts", "fav_roll5_pts",
  "weather_temperature", "to_momentum_diff", "success_rate_diff",
  "qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
  "third_down_epa_diff", "redzone_epa_diff",
  "pass_epa_mismatch", "rush_epa_mismatch",
  "explosive_diff", "neg_play_diff",
  "off_epa_diff", "def_epa_diff",
  "injury_diff", "fav_qb_out", "dog_qb_out",
  "div_game", "is_primetime", "is_dome",
  "rest_diff", "fav_at_home", "dog_short_week",
  "stadium_neutral"
)

# ============================================================
# WALK-FORWARD MARGIN REGRESSION
# ============================================================
cat("\n=== WALK-FORWARD MARGIN REGRESSION ===\n\n")

# Ridge regression (continuous target)
ridge_reg_spec <- linear_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("regression")

ridge_reg_grid <- grid_regular(penalty(range = c(-4, 1)), levels = 25)

margin_results <- list()

for (test_s in c(2023, 2024, 2025)) {
  cat("Test season", test_s, ":\n")

  train <- margin_data |> filter(season < test_s) |> select(any_of(reg_vars))
  test <- margin_data |> filter(season == test_s) |> select(any_of(reg_vars))

  # Temporal CV
  inner_cv <- make_temporal_cv(train, min_train_season = 2012)

  recipe_margin <- recipe(margin_vs_spread ~ ., data = train) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

  wf <- workflow() |> add_model(ridge_reg_spec) |> add_recipe(recipe_margin)

  set.seed(301)
  tune_result <- tune_grid(wf, resamples = inner_cv, grid = ridge_reg_grid,
                           metrics = metric_set(rmse, mae, rsq),
                           control = control_grid(save_pred = FALSE, verbose = FALSE))

  best <- select_best(tune_result, metric = "rmse")

  # Fit and predict
  final_fit <- fit(finalize_workflow(wf, best), data = train)
  test_preds <- test |>
    select(game_id_norm, season, week, margin_vs_spread) |>
    bind_cols(predict(final_fit, new_data = test))

  # Estimate prediction uncertainty from training residuals
  train_preds <- predict(final_fit, new_data = train)
  train_residuals <- train$margin_vs_spread - train_preds$.pred
  residual_sd <- sd(train_residuals)

  test_preds <- test_preds |>
    mutate(
      pred_margin = .pred,
      residual_sd = residual_sd,
      # Probability that margin_vs_spread > 0 (i.e., covers)
      p_cover = pnorm(0, mean = pred_margin, sd = residual_sd, lower.tail = FALSE),
      # Probability that margin_vs_spread > 0 enough to overcome vig
      # At -110, need ~52.4% edge
      p_profitable = pnorm(0, mean = pred_margin, sd = residual_sd, lower.tail = FALSE),
      # Actual cover
      actual_cover = margin_vs_spread > 0,
      # Betting decision: only bet when model predicts |margin| > threshold
      pred_side = if_else(pred_margin > 0, "cover", "no_cover")
    )

  margin_results[[as.character(test_s)]] <- test_preds

  # Metrics
  rmse_val <- sqrt(mean((test_preds$margin_vs_spread - test_preds$pred_margin)^2))
  mae_val <- mean(abs(test_preds$margin_vs_spread - test_preds$pred_margin))
  cor_val <- cor(test_preds$margin_vs_spread, test_preds$pred_margin)
  direction_acc <- mean((test_preds$pred_margin > 0) == test_preds$actual_cover)

  cat("  RMSE:", round(rmse_val, 2), "| MAE:", round(mae_val, 2),
      "| Cor:", round(cor_val, 4), "| Direction acc:", round(direction_acc, 3), "\n")
}

margin_all <- bind_rows(margin_results)

# ============================================================
# MARGIN-BASED BETTING STRATEGY
# ============================================================
cat("\n=== MARGIN-BASED BETTING ===\n")
cat("Only bet when predicted margin exceeds threshold\n\n")

for (thresh in c(0, 0.5, 1, 1.5, 2, 3, 4, 5)) {
  bets <- margin_all |>
    filter(abs(pred_margin) > thresh) |>
    mutate(
      correct = (pred_margin > 0) == actual_cover,
      profit = if_else(correct, 100, -110)
    )

  if (nrow(bets) < 20) next

  win_rate <- mean(bets$correct)
  se <- sqrt(win_rate * (1 - win_rate) / nrow(bets))

  cat(sprintf("  |margin| > %3.1f: %4d bets | win: %.1f%% +/- %.1f%% | profit: $%6.0f | ROI: %5.1f%%\n",
              thresh, nrow(bets), win_rate * 100, 1.645 * se * 100,
              sum(bets$profit), sum(bets$profit) / (nrow(bets) * 110) * 100))
}

# ============================================================
# DISTRIBUTIONAL ANALYSIS
# ============================================================
cat("\n=== DISTRIBUTIONAL ANALYSIS ===\n")
cat("Using residual SD to estimate P(cover) from predicted margin\n\n")

# For each game, compute P(cover | model) and bet when confident
margin_all <- margin_all |>
  mutate(
    ev_cover = p_cover * 100 - (1 - p_cover) * 110,
    ev_nocover = (1 - p_cover) * 100 - p_cover * 110,
    best_ev = pmax(ev_cover, ev_nocover),
    best_side = if_else(ev_cover >= ev_nocover, "cover", "no_cover"),
    bet_correct = best_side == if_else(actual_cover, "cover", "no_cover"),
    profit = if_else(bet_correct, 100, -110)
  )

cat("Distributional EV betting:\n")
for (ev_thresh in c(0, 1, 2, 3, 5)) {
  bets <- margin_all |> filter(best_ev > ev_thresh)
  if (nrow(bets) < 20) next

  cat(sprintf("  EV > $%d: %4d bets | win: %.1f%% | profit: $%6.0f | ROI: %5.1f%%\n",
              ev_thresh, nrow(bets), mean(bets$bet_correct) * 100,
              sum(bets$profit), sum(bets$profit) / (nrow(bets) * 110) * 100))
}

# ============================================================
# KEY NUMBERS ANALYSIS
# ============================================================
cat("\n=== KEY NUMBERS: Does crossing key numbers matter? ===\n")

margin_all <- margin_all |>
  mutate(
    crosses_3 = (pred_margin > 0 & spread_fav > -3) | (pred_margin < 0 & spread_fav < -3),
    crosses_7 = (pred_margin > 0 & spread_fav > -7) | (pred_margin < 0 & spread_fav < -7)
  )

# Save
write_rds(margin_all, here("objects", "margin_model_predictions.rds"))

cat("\n=== SEASON BREAKDOWN ===\n")
margin_all |>
  group_by(season) |>
  summarize(
    n = n(),
    rmse = round(sqrt(mean((margin_vs_spread - pred_margin)^2)), 2),
    cor = round(cor(margin_vs_spread, pred_margin), 4),
    direction_acc = round(mean((pred_margin > 0) == actual_cover), 3),
    .groups = "drop"
  ) |> print()

cat("\nDone!\n")
