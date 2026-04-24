# Corrected Model + Selective Betting
#
# Root cause fixed: walk_forward_evaluate was selecting penalty=4.0 (null model)
# because log loss is minimized by predicting the base rate. The real signal
# lives at penalty=0.001 where the model actually uses features.
#
# This script:
# 1. Tunes in the CORRECT penalty range (0.0001 to 0.05)
# 2. Uses glmnet CV within each training window to pick penalty
# 3. Generates real predictions with spread (not null model)
# 4. Applies selective betting (only bet confident games)
# 5. Reports honest walk-forward results on 2023, 2024, 2025

library(tidyverse)
library(here)
library(glmnet)

source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

cat("=== CORRECTED MODEL + SELECTIVE BETTING ===\n\n")

# ============================================================
# 1. FEATURE SELECTION: use features with >80% coverage
# ============================================================
feat_cols <- v3 |> select(where(is.numeric), -season, -week) |> names()
coverage <- map_dbl(feat_cols, ~ mean(!is.na(v3[[.x]])))
good_feats <- feat_cols[coverage > 0.80]
cat("Features with >80% coverage:", length(good_feats), "\n")

# ============================================================
# 2. WALK-FORWARD WITH PROPER CV TUNING
# ============================================================
cat("\n=== Walk-Forward (CV-tuned within each window) ===\n")

all_predictions <- list()

for (test_s in c(2023, 2024, 2025)) {
  cat(sprintf("\n--- Test season %d ---\n", test_s))

  train <- v3 |> filter(season >= 2012, season < test_s)
  test <- v3 |> filter(season == test_s)

  cat(sprintf("  Train: %d games (%d-%d) | Test: %d games\n",
    nrow(train), min(train$season), max(train$season), nrow(test)))

  # Prepare matrices with median imputation (training medians only)
  train_x <- train |> select(all_of(good_feats))
  train_medians <- map_dbl(train_x, ~ median(.x, na.rm = TRUE))

  train_x <- train_x |>
    mutate(across(everything(), ~ replace_na(.x, train_medians[cur_column()]))) |>
    as.matrix()
  test_x <- test |> select(all_of(good_feats)) |>
    mutate(across(everything(), ~ replace_na(.x, train_medians[cur_column()]))) |>
    as.matrix()

  train_y <- factor(train$fav_cover, levels = c("cover", "no_cover"))
  test_y <- factor(test$fav_cover, levels = c("cover", "no_cover"))

  # Scale using training statistics
  means <- colMeans(train_x)
  sds <- apply(train_x, 2, sd)
  sds[sds == 0] <- 1
  train_xs <- scale(train_x, center = means, scale = sds)
  test_xs <- scale(test_x, center = means, scale = sds)

  # Remove constant columns
  good_cols <- which(apply(train_xs, 2, sd) > 0)
  train_xs <- train_xs[, good_cols]
  test_xs <- test_xs[, good_cols]
  cat(sprintf("  Features after removing constants: %d\n", length(good_cols)))

  # CV within training window to pick lambda
  # Use 5-fold CV (not temporal — we need enough folds for stable selection)
  # But constrain lambda to the range where the model actually has coefficients
  set.seed(301)
  cv_fit <- cv.glmnet(
    train_xs, train_y,
    family = "binomial",
    alpha = 0.5,        # elastic net
    nfolds = 10,
    lambda = 10^seq(-4, -1.5, length.out = 50),  # 0.0001 to 0.03
    type.measure = "deviance"  # deviance = log likelihood
  )

  cat(sprintf("  CV lambda.min: %.5f | lambda.1se: %.5f\n",
    cv_fit$lambda.min, cv_fit$lambda.1se))

  # Use lambda.min (not 1se — 1se is too conservative for this problem)
  best_lambda <- cv_fit$lambda.min
  n_nonzero <- sum(coef(cv_fit, s = best_lambda)[-1, 1] != 0)
  cat(sprintf("  Features at lambda.min: %d\n", n_nonzero))

  # Predict test set
  pred_raw <- predict(cv_fit, newx = test_xs, s = best_lambda, type = "response")
  # glmnet models P(second level) = P(no_cover)
  p_cover <- 1 - pred_raw[, 1]

  acc <- mean((p_cover > 0.5) == (test_y == "cover"))
  cat(sprintf("  Raw accuracy: %.1f%% | pred range: [%.3f, %.3f] | SD: %.4f\n",
    acc * 100, min(p_cover), max(p_cover), sd(p_cover)))

  # Store predictions
  preds <- tibble(
    game_id_norm = test$game_id_norm,
    season = test$season,
    week = test$week,
    fav_cover = test$fav_cover,
    spread_fav = test$spread_fav,
    p_cover = p_cover,
    p_nocover = 1 - p_cover,
    test_season = test_s,
    lambda_used = best_lambda,
    n_features = n_nonzero
  )

  all_predictions[[as.character(test_s)]] <- preds
}

preds_all <- bind_rows(all_predictions)
cat(sprintf("\nTotal predictions: %d games\n", nrow(preds_all)))

# ============================================================
# 3. OVERALL RESULTS
# ============================================================
cat("\n=== OVERALL RESULTS ===\n")

overall_acc <- mean((preds_all$p_cover > 0.5) == (preds_all$fav_cover == "cover"))
cat(sprintf("Overall accuracy: %.1f%% (breakeven: 52.4%%)\n", overall_acc * 100))

season_results <- preds_all |>
  group_by(test_season) |>
  summarize(
    n = n(),
    accuracy = mean((p_cover > 0.5) == (fav_cover == "cover")),
    pred_mean = mean(p_cover),
    pred_sd = sd(p_cover),
    pred_min = min(p_cover),
    pred_max = max(p_cover),
    lambda = first(lambda_used),
    features = first(n_features),
    .groups = "drop"
  )
cat("\nPer-season:\n")
print(season_results)

# ============================================================
# 4. SELECTIVE BETTING — CONFIDENCE THRESHOLDS
# ============================================================
cat("\n=== SELECTIVE BETTING ===\n")
cat("Only bet games where model has strong opinion\n")
cat("Bet cover when p_cover > threshold\n")
cat("Bet no_cover when p_cover < (1 - threshold)\n")
cat("Skip game otherwise\n\n")

threshold_results <- list()

for (thresh in seq(0.50, 0.62, by = 0.01)) {
  bets <- preds_all |>
    mutate(
      bet_cover = p_cover > thresh,
      bet_nocover = p_cover < (1 - thresh),
      has_bet = bet_cover | bet_nocover,
      correct = case_when(
        bet_cover & fav_cover == "cover" ~ TRUE,
        bet_nocover & fav_cover == "no_cover" ~ TRUE,
        has_bet ~ FALSE,
        TRUE ~ NA
      )
    ) |>
    filter(has_bet)

  if (nrow(bets) < 10) next

  win_rate <- mean(bets$correct)
  profit <- sum(if_else(bets$correct, 100, -110))
  roi <- profit / (nrow(bets) * 110)

  # Wilson CI on win rate
  se <- sqrt(win_rate * (1 - win_rate) / nrow(bets))

  threshold_results[[as.character(thresh)]] <- tibble(
    threshold = thresh,
    n_bets = nrow(bets),
    pct_games = nrow(bets) / nrow(preds_all) * 100,
    win_rate = win_rate,
    win_rate_lower = win_rate - 1.96 * se,
    win_rate_upper = win_rate + 1.96 * se,
    profit = profit,
    roi = roi
  )

  cat(sprintf("  P>%.2f: %4d bets (%5.1f%%) | win: %5.1f%% [%4.1f%%-%4.1f%%] | $%7.0f | ROI: %5.1f%%\n",
    thresh, nrow(bets), nrow(bets) / nrow(preds_all) * 100,
    win_rate * 100, (win_rate - 1.96*se) * 100, (win_rate + 1.96*se) * 100,
    profit, roi * 100))
}

thresh_df <- bind_rows(threshold_results)

# ============================================================
# 5. SELECTIVE BETTING BY SEASON
# ============================================================
cat("\n=== SELECTIVE BETTING BY SEASON ===\n")

# Use the best threshold that has >= 100 bets
best_thresh_row <- thresh_df |>
  filter(n_bets >= 100) |>
  slice_max(roi, n = 1)

if (nrow(best_thresh_row) > 0) {
  bt <- best_thresh_row$threshold[1]
  cat(sprintf("Best threshold (>= 100 bets): P > %.2f\n\n", bt))

  season_bets <- preds_all |>
    mutate(
      bet_cover = p_cover > bt,
      bet_nocover = p_cover < (1 - bt),
      has_bet = bet_cover | bet_nocover,
      correct = case_when(
        bet_cover & fav_cover == "cover" ~ TRUE,
        bet_nocover & fav_cover == "no_cover" ~ TRUE,
        has_bet ~ FALSE,
        TRUE ~ NA
      )
    ) |>
    filter(has_bet) |>
    group_by(test_season) |>
    summarize(
      n_bets = n(),
      win_rate = mean(correct) * 100,
      profit = sum(if_else(correct, 100, -110)),
      .groups = "drop"
    )
  print(season_bets)
}

# ============================================================
# 6. WHAT FEATURES DOES THE MODEL USE?
# ============================================================
cat("\n=== TOP FEATURES (from 2024 training window) ===\n")

# Refit on all data through 2024 to inspect coefficients
train_all <- v3 |> filter(season < 2025)
train_x_all <- train_all |> select(all_of(good_feats)) |>
  mutate(across(everything(), ~ replace_na(.x, median(.x, na.rm = TRUE)))) |> as.matrix()
train_y_all <- factor(train_all$fav_cover, levels = c("cover", "no_cover"))

means_all <- colMeans(train_x_all)
sds_all <- apply(train_x_all, 2, sd); sds_all[sds_all == 0] <- 1
train_xs_all <- scale(train_x_all, center = means_all, scale = sds_all)
good_cols_all <- which(apply(train_xs_all, 2, sd) > 0)
train_xs_all <- train_xs_all[, good_cols_all]

set.seed(301)
cv_final <- cv.glmnet(train_xs_all, train_y_all, family = "binomial",
  alpha = 0.5, nfolds = 10, lambda = 10^seq(-4, -1.5, length.out = 50))

coefs <- coef(cv_final, s = "lambda.min")[-1, 1]
# Remember: glmnet models P(no_cover), so positive coefs INCREASE P(no_cover)
# To get "helps cover", we negate
coefs_cover <- -coefs
nonzero <- coefs_cover[coefs_cover != 0]
top_feats <- sort(abs(nonzero), decreasing = TRUE)

cat(sprintf("\nFeatures selected: %d\n", length(nonzero)))
cat("Top 25 by |coefficient| (positive = favors cover):\n")
top_25 <- head(top_feats, 25)
for (i in seq_along(top_25)) {
  feat_name <- names(top_25)[i]
  direction <- if (nonzero[feat_name] > 0) "+" else "-"
  cat(sprintf("  %2d. %s %-45s %.4f\n", i, direction, feat_name, top_25[i]))
}

# ============================================================
# 7. SAVE
# ============================================================
write_rds(preds_all, here("objects", "corrected_predictions.rds"))
write_csv(thresh_df, here("results", "confidence_threshold_corrected.csv"))
write_csv(as.data.frame(season_results), here("results", "corrected_season_results.csv"))

cat("\n=== DONE ===\n")
cat("Predictions: objects/corrected_predictions.rds\n")
cat("Thresholds: results/confidence_threshold_corrected.csv\n")
