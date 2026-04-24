# Era-3 Model (2022+ with FTN/NGS) + Recent-Only Training
#
# Key insight: when we train on 2012+, FTN/NGS features are NA for 60% of rows.
# Median imputation fills them with fake values. By training ONLY on 2022+,
# every row has REAL premium feature values. The model can actually learn from:
# - fav_ftn_roll5_rpo (cor=-0.067, strongest premium signal)
# - fav_ftn_roll5_blitz (cor=0.052)
# - fav_ngs_aggressiveness_roll5 (cor=0.053)

library(tidyverse)
library(here)
library(glmnet)

source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

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

# ALL features including FTN (29% coverage overall but 100% in 2022+)
all_feats <- v3 |> select(where(is.numeric), -season, -week) |> names()
stable_feats <- setdiff(all_feats, flip_features)

# ============================================================
# Helper: run model with specified train window
# ============================================================
run_model <- function(train_start, test_season, feats, alpha = 0) {
  train <- v3 |> filter(season >= train_start, season < test_season)
  test <- v3 |> filter(season == test_season)
  if (nrow(train) < 50) return(NULL)

  # For era-3, only keep features with >50% coverage in the TRAINING window
  train_coverage <- map_dbl(feats, ~ mean(!is.na(train[[.x]])))
  era_feats <- feats[train_coverage > 0.50]

  train_x <- train |> select(all_of(era_feats))
  meds <- map_dbl(train_x, ~ median(.x, na.rm = TRUE))
  train_x <- train_x |> mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()
  test_x <- test |> select(all_of(era_feats)) |>
    mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()

  m <- colMeans(train_x); s <- apply(train_x, 2, sd); s[s == 0] <- 1
  train_xs <- scale(train_x, m, s); test_xs <- scale(test_x, m, s)
  gc <- which(apply(train_xs, 2, sd) > 0)
  if (length(gc) < 3) return(NULL)
  train_xs <- train_xs[, gc]; test_xs <- test_xs[, gc]

  y <- factor(train$fav_cover, levels = c("cover", "no_cover"))

  set.seed(301)
  nfolds <- min(10, max(3, nrow(train) %/% 30))
  cv_fit <- tryCatch(
    cv.glmnet(train_xs, y, family = "binomial", alpha = alpha, nfolds = nfolds,
               lambda = 10^seq(-3.5, -0.3, length.out = 40)),
    error = function(e) NULL
  )
  if (is.null(cv_fit)) return(NULL)

  pred <- predict(cv_fit, test_xs, s = "lambda.min", type = "response")[, 1]
  p_cover <- 1 - pred

  tibble(
    game_id_norm = test$game_id_norm,
    fav_cover = test$fav_cover,
    p_cover = p_cover,
    test_season = test_season,
    train_start = train_start,
    model_type = paste0("train_", train_start, "_alpha_", alpha),
    n_features = length(gc),
    lambda = cv_fit$lambda.min
  )
}

# ============================================================
# Model A: Era-3 (train 2022+ only) — FTN/NGS fully available
# ============================================================
cat("=== MODEL A: Era-3 (2022+ training, all premium features) ===\n")

era3_results <- list()
for (test_s in c(2023, 2024, 2025)) {
  res <- run_model(2022, test_s, stable_feats, alpha = 0)
  if (!is.null(res)) {
    era3_results[[as.character(test_s)]] <- res
    acc <- mean((res$p_cover > 0.5) == (res$fav_cover == "cover"))
    cover_bets <- sum(res$p_cover > 0.54)
    cover_wr <- if (cover_bets >= 5) mean((res$fav_cover == "cover")[res$p_cover > 0.54]) else NA
    cat(sprintf("  %d: acc=%.1f%%, %d features, cover P>0.54: %d bets, %.1f%%\n",
      test_s, acc * 100, res$n_features[1], cover_bets,
      ifelse(is.na(cover_wr), 0, cover_wr * 100)))
  }
}

# ============================================================
# Model B: Recent windows (2-6 seasons back)
# ============================================================
cat("\n=== MODEL B: Recent window sweep ===\n")

recent_results <- list()
for (window_size in c(2, 3, 4, 5, 6)) {
  cat(sprintf("\nWindow = %d seasons:\n", window_size))
  for (test_s in c(2023, 2024, 2025)) {
    start <- test_s - window_size
    res <- run_model(start, test_s, stable_feats, alpha = 0)
    if (!is.null(res)) {
      res$window_size <- window_size
      recent_results[[paste(window_size, test_s)]] <- res
      acc <- mean((res$p_cover > 0.5) == (res$fav_cover == "cover"))
      cover_bets <- sum(res$p_cover > 0.54)
      cover_wr <- if (cover_bets >= 5) mean((res$fav_cover == "cover")[res$p_cover > 0.54]) else NA
      cat(sprintf("  %d (train %d-%d): acc=%.1f%%, cover: %d bets, %.1f%%\n",
        test_s, start, test_s - 1, acc * 100, cover_bets,
        ifelse(is.na(cover_wr), 0, cover_wr * 100)))
    }
  }
}

# ============================================================
# Compare all models
# ============================================================
cat("\n=== COMPARISON ===\n")

# Full history model (from existing predictions)
full_p <- read_rds(here("objects", "tuned_stable_predictions.rds"))

# Era-3 combined
era3_p <- bind_rows(era3_results)

# Best recent window
all_recent <- bind_rows(recent_results)

cat("\n--- Full history (2012+) ---\n")
for (test_s in c(2023, 2024, 2025)) {
  sub <- full_p |> filter(test_season == test_s)
  cover <- sub |> filter(p_cover > 0.54)
  if (nrow(cover) >= 5) {
    cat(sprintf("  %d: %d cover bets, %.1f%%\n", test_s, nrow(cover), mean(cover$truth_cover) * 100))
  }
}

if (nrow(era3_p) > 0) {
  cat("\n--- Era-3 (2022+ training) ---\n")
  for (test_s in unique(era3_p$test_season)) {
    sub <- era3_p |> filter(test_season == test_s)
    cover <- sub |> filter(p_cover > 0.54)
    if (nrow(cover) >= 5) {
      cat(sprintf("  %d: %d cover bets, %.1f%%\n", test_s, nrow(cover),
        mean((cover$fav_cover == "cover")) * 100))
    }
  }
}

# Summary table by window size
cat("\n--- By window size (cover-side P > 0.54) ---\n")
all_recent |>
  mutate(truth_cover = fav_cover == "cover") |>
  filter(p_cover > 0.54) |>
  group_by(window_size) |>
  summarize(
    n_bets = n(),
    win_rate = round(mean(truth_cover) * 100, 1),
    profit = sum(ifelse(truth_cover, 100, -110)),
    roi = round(sum(ifelse(truth_cover, 100, -110)) / (n() * 110) * 100, 1),
    .groups = "drop"
  ) |> print()

# Save
write_rds(era3_p, here("objects", "era3_predictions.rds"))
write_rds(all_recent, here("objects", "recent_window_predictions.rds"))

cat("\nDone.\n")
