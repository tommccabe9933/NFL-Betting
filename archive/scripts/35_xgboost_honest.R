# XGBoost with Honest Tuning
# Tune on FULL logloss (not bet-filtered accuracy)
# Threshold applied AFTER tuning, not during
# 3-season rolling window, same as our best ridge

library(tidyverse)
library(here)
library(xgboost)

source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

flip_features <- c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff","rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa","injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged","fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3","fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3","off_success_rate_roll5_diff","def_pass_epa_roll3_diff","fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff","fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5","fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5","dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff","off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff","def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
stable_feats <- setdiff(v3 |> select(where(is.numeric), -season, -week) |> names(), flip_features)

cat("=== XGBOOST WITH HONEST TUNING ===\n")
cat("Features:", length(stable_feats), "\n\n")

# ============================================================
# Grid search: tune on full logloss via xgb.cv
# ============================================================

# Param grid — reduced to ~48 combos (tractable on laptop)
param_grid <- expand.grid(
  max_depth = c(2, 3, 4),
  eta = c(0.01, 0.05, 0.10),
  subsample = c(0.7, 0.85),
  colsample_bytree = c(0.5, 0.7),
  min_child_weight = c(20, 50),
  gamma = c(0, 1)
)
cat("Param combos:", nrow(param_grid), "\n")

all_preds <- list()
all_importance <- list()

for (test_s in c(2023, 2024, 2025)) {
  cat(sprintf("\n--- Test season %d ---\n", test_s))

  train_start <- test_s - 3
  train <- v3 |> filter(season >= train_start, season < test_s)
  test <- v3 |> filter(season == test_s)

  # Feature coverage
  tc <- map_dbl(stable_feats, ~ mean(!is.na(train[[.x]])))
  feats <- stable_feats[tc > 0.50]

  # Prep matrices
  train_x <- train |> select(all_of(feats))
  meds <- map_dbl(train_x, ~ median(.x, na.rm = TRUE))
  train_x <- train_x |> mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()
  test_x <- test |> select(all_of(feats)) |>
    mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()

  # XGBoost labels: 1 = cover, 0 = no_cover
  train_y <- as.integer(train$fav_cover == "cover")
  test_y <- as.integer(test$fav_cover == "cover")

  dtrain <- xgb.DMatrix(data = train_x, label = train_y)
  dtest <- xgb.DMatrix(data = test_x, label = test_y)

  # Grid search with xgb.cv (tune on FULL logloss)
  best_logloss <- Inf
  best_params <- NULL
  best_nrounds <- NULL

  for (i in seq_len(nrow(param_grid))) {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = param_grid$max_depth[i],
      eta = param_grid$eta[i],
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i],
      min_child_weight = param_grid$min_child_weight[i],
      gamma = param_grid$gamma[i]
    )

    set.seed(301)
    cv_result <- tryCatch(
      xgb.cv(
        params = params,
        data = dtrain,
        nrounds = 500,
        nfold = 5,
        early_stopping_rounds = 20,
        verbose = 0
      ),
      error = function(e) NULL
    )

    if (is.null(cv_result)) next

    cv_logloss <- min(cv_result$evaluation_log$test_logloss_mean)
    cv_nrounds <- cv_result$best_iteration
    if (is.null(cv_nrounds) || cv_nrounds < 5) next  # skip degenerate fits

    if (cv_logloss < best_logloss) {
      best_logloss <- cv_logloss
      best_params <- params
      best_nrounds <- cv_nrounds
    }
  }

  if (is.null(best_nrounds)) {
    cat(sprintf("  No valid XGBoost config found for %d — skipping\n", test_s))
    next
  }
  cat(sprintf("  Best CV logloss: %.5f at %d rounds\n", best_logloss, best_nrounds))
  cat(sprintf("  Params: depth=%d eta=%.2f sub=%.2f col=%.2f mcw=%d gamma=%d\n",
    best_params$max_depth, best_params$eta, best_params$subsample,
    best_params$colsample_bytree, best_params$min_child_weight, best_params$gamma))

  # Train final model with best params
  set.seed(301)
  final_model <- xgb.train(
    params = best_params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )

  # Predict test set — XGBoost outputs P(cover) directly since label=1 is cover
  p_cover <- predict(final_model, dtest)

  overall_acc <- mean((p_cover > 0.5) == (test_y == 1))
  cat(sprintf("  Overall accuracy: %.1f%%\n", overall_acc * 100))
  cat(sprintf("  Pred range: [%.3f, %.3f], SD: %.4f\n", min(p_cover), max(p_cover), sd(p_cover)))

  # Store predictions
  all_preds[[as.character(test_s)]] <- tibble(
    game_id_norm = test$game_id_norm,
    fav_cover = test$fav_cover,
    p_cover = p_cover,
    test_season = test_s,
    truth_cover = test_y == 1
  )

  # Feature importance
  imp <- xgb.importance(model = final_model)
  all_importance[[as.character(test_s)]] <- imp |> mutate(test_season = test_s)
}

# ============================================================
# Combine and analyze
# ============================================================
preds <- bind_rows(all_preds)
cat(sprintf("\nTotal XGBoost predictions: %d games\n", nrow(preds)))

# Cover-side threshold sweep (applied AFTER tuning)
cat("\n=== COVER-SIDE THRESHOLD SWEEP ===\n")
thresh_results <- list()
for (th in seq(0.52, 0.60, by = 0.01)) {
  bets <- preds |> filter(p_cover > th)
  if (nrow(bets) < 5) next
  wr <- mean(bets$truth_cover)
  profit <- sum(ifelse(bets$truth_cover, 100, -110))
  se <- sqrt(wr * (1 - wr) / nrow(bets))
  thresh_results[[as.character(th)]] <- tibble(threshold = th, n = nrow(bets),
    win_rate = wr, profit = profit, roi = profit / (nrow(bets) * 110))
  cat(sprintf("  P>%.2f: %4d bets | win: %5.1f%% [%4.1f%%-%4.1f%%] | $%6.0f | ROI: %5.1f%%\n",
    th, nrow(bets), wr * 100, (wr - 1.96 * se) * 100, (wr + 1.96 * se) * 100,
    profit, profit / (nrow(bets) * 110) * 100))
}

# Per season at P > 0.55
cat("\n=== BY SEASON (P > 0.55) ===\n")
preds |> filter(p_cover > 0.55) |>
  group_by(test_season) |>
  summarize(n = n(), wr = round(mean(truth_cover) * 100, 1),
            profit = sum(ifelse(truth_cover, 100, -110)), .groups = "drop") |>
  print()

# Feature importance (averaged across windows)
cat("\n=== TOP 20 FEATURES (avg gain across seasons) ===\n")
all_imp <- bind_rows(all_importance)
avg_imp <- all_imp |>
  group_by(Feature) |>
  summarize(avg_gain = mean(Gain), n_seasons = n(), .groups = "drop") |>
  arrange(desc(avg_gain))
avg_imp |> head(20) |> print()

# Compare to ridge
cat("\n=== COMPARE: XGBoost vs Ridge ===\n")
cat("(Ridge best: train=2022 a=0.1 lam=0.005 th=0.57: 332 bets, 57.8%, $3800)\n\n")

write_rds(preds, here("objects", "xgb_predictions.rds"))
write_csv(bind_rows(thresh_results), here("results", "xgb_threshold_sweep.csv"))
write_csv(avg_imp, here("results", "xgb_feature_importance.csv"))

cat("\nDone.\n")
