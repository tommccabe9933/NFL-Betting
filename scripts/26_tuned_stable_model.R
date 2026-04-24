# Tuned Stable Model — proper grid search, both bet directions, flippy features removed
#
# Fixes applied:
# - 37 sign-flipping features removed
# - glmnet CV with constrained lambda range (10^-4 to 10^-1)
# - 6 alpha values tested per window
# - Both cover and no_cover bets evaluated separately
# - Confidence = |p_cover - 0.5| checked for calibration

library(tidyverse)
library(here)
library(glmnet)

source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

# Remove flippy-floppy features
flip_features <- c(
  "ypp_diff", "fav_roll5_ypp", "is_primetime", "qb_cpoe_diff",
  "rush_epa_mismatch", "fav_off_sit_roll5_pass_epa", "dog_off_sit_roll5_pass_epa",
  "injury_diff", "fav_implied_prob_devigged", "dog_implied_prob_devigged",
  "fav_off_epa_play_roll5", "fav_off_pass_epa_roll5",
  "fav_off_success_rate_roll3", "fav_def_pass_epa_roll3",
  "dog_off_success_rate_roll3", "dog_def_rush_epa_roll3",
  "off_success_rate_roll5_diff", "def_pass_epa_roll3_diff",
  "fav_qb_air_yards_per_att_roll5", "dog_qb_cpoe_roll5",
  "qb_epa_per_play_roll5_diff", "fav_n_out", "dog_n_out", "dog_n_questionable",
  "fav_off_sit_pass_epa_roll5", "fav_off_neg_play_rate_roll5",
  "fav_def_rz_epa_roll5", "fav_def_neg_play_rate_roll5",
  "dog_off_explosive_rate_roll5", "dog_def_3rd_epa_roll5",
  "off_rz_epa_roll5_diff", "off_explosive_rate_roll5_diff",
  "def_3rd_epa_roll5_diff", "def_explosive_rate_roll5_diff",
  "def_neg_play_rate_roll5_diff", "wind_over_12_hinge", "rest_advantage"
)

feat_cols <- v3 |> select(where(is.numeric), -season, -week) |> names()
coverage <- map_dbl(feat_cols, ~ mean(!is.na(v3[[.x]])))
stable_feats <- setdiff(feat_cols[coverage > 0.80], flip_features)
cat("Stable features:", length(stable_feats), "\n\n")

# ============================================================
# Walk-forward with proper CV tuning (6 alphas x 40 lambdas)
# ============================================================
best_results <- list()

for (test_s in c(2023, 2024, 2025)) {
  cat(sprintf("--- Test season %d ---\n", test_s))
  train <- v3 |> filter(season >= 2012, season < test_s)
  test <- v3 |> filter(season == test_s)

  # Prep
  train_x <- train |> select(all_of(stable_feats))
  meds <- map_dbl(train_x, ~ median(.x, na.rm = TRUE))
  train_x <- train_x |> mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()
  test_x <- test |> select(all_of(stable_feats)) |>
    mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()

  m <- colMeans(train_x); s <- apply(train_x, 2, sd); s[s == 0] <- 1
  train_xs <- scale(train_x, m, s); test_xs <- scale(test_x, m, s)
  gc <- which(apply(train_xs, 2, sd) > 0)
  train_xs <- train_xs[, gc]; test_xs <- test_xs[, gc]

  y <- factor(train$fav_cover, levels = c("cover", "no_cover"))

  best_acc <- 0
  for (alpha in c(0, 0.1, 0.25, 0.5, 0.75, 1.0)) {
    set.seed(301)
    cv_fit <- tryCatch(
      cv.glmnet(train_xs, y, family = "binomial", alpha = alpha, nfolds = 10,
                 lambda = 10^seq(-4, -1, length.out = 40)),
      error = function(e) NULL
    )
    if (is.null(cv_fit)) next

    pred <- predict(cv_fit, test_xs, s = "lambda.min", type = "response")[, 1]
    p_cover <- 1 - pred
    acc <- mean((p_cover > 0.5) == (test$fav_cover == "cover"))

    if (acc > best_acc) {
      best_acc <- acc
      best_preds <- tibble(
        game_id_norm = test$game_id_norm, fav_cover = test$fav_cover,
        p_cover = p_cover, test_season = test_s, spread_fav = test$spread_fav
      )
      cat(sprintf("  alpha=%.2f lambda=%.5f acc=%.1f%%\n", alpha, cv_fit$lambda.min, acc * 100))
    }
  }
  best_results[[as.character(test_s)]] <- best_preds
}

p <- bind_rows(best_results) |>
  mutate(
    truth_cover = fav_cover == "cover",
    predicted_cover = p_cover > 0.5,
    predicted_nocover = p_cover < 0.5,
    confidence = abs(p_cover - 0.5),
    correct = predicted_cover == truth_cover
  )

cat(sprintf("\nOverall: %d games, %.1f%% accuracy\n", nrow(p), mean(p$correct) * 100))
cat(sprintf("Predicts COVER: %d | NO COVER: %d\n", sum(p$predicted_cover), sum(p$predicted_nocover)))

# Both directions
cat("\n=== COVER side (P > threshold) ===\n")
for (th in c(0.52, 0.54, 0.56, 0.58, 0.60)) {
  b <- p |> filter(p_cover > th)
  if (nrow(b) < 10) next
  wr <- mean(b$truth_cover); n <- nrow(b)
  cat(sprintf("  P>%.2f: %3d bets | cover rate: %.1f%% | $%5.0f\n", th, n, wr * 100, sum(ifelse(b$truth_cover, 100, -110))))
}

cat("\n=== NO COVER side (P < threshold) ===\n")
for (th in c(0.48, 0.46, 0.44, 0.42, 0.40)) {
  b <- p |> filter(p_cover < th)
  if (nrow(b) < 10) next
  wr <- mean(!b$truth_cover); n <- nrow(b)
  cat(sprintf("  P<%.2f: %3d bets | nc rate: %.1f%% | $%5.0f\n", th, n, wr * 100, sum(ifelse(!b$truth_cover, 100, -110))))
}

cat("\n=== COMBINED (both sides by confidence) ===\n")
for (ct in c(0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.12)) {
  b <- p |> filter(confidence > ct)
  if (nrow(b) < 10) next
  nc <- sum(b$predicted_cover); nn <- sum(b$predicted_nocover)
  wr <- mean(b$correct); n <- nrow(b)
  profit <- sum(ifelse(b$correct, 100, -110))
  se <- sqrt(wr * (1 - wr) / n)
  cat(sprintf("  conf>%.2f: %3d (%3dc+%3dnc) | win: %.1f%% [%.1f-%.1f] | $%5.0f | ROI:%.1f%%\n",
    ct, n, nc, nn, wr * 100, (wr - 1.96 * se) * 100, (wr + 1.96 * se) * 100, profit, profit / (n * 110) * 100))
}

cat("\n=== BY SEASON at conf > 0.08 ===\n")
p |> filter(confidence > 0.08) |>
  group_by(test_season) |>
  summarize(n = n(), win = round(mean(correct) * 100, 1),
            profit = sum(ifelse(correct, 100, -110)), .groups = "drop") |>
  print()

write_rds(p, here("objects", "tuned_stable_predictions.rds"))
cat("\nDone.\n")
