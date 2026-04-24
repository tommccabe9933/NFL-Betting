# Diagnose 2025 + Training Window Sweep
# Why did 2025 fail on cover-side? Can shorter training windows fix it?

library(tidyverse)
library(here)
library(glmnet)

source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))
p <- read_rds(here("objects", "tuned_stable_predictions.rds"))

# Flippy features to remove
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

# ============================================================
# 1. Was 2025 a weird year overall?
# ============================================================
cat("=== 2025 BASELINE CHECK ===\n")
season_cover_rates <- v3 |>
  group_by(season) |>
  summarize(n = n(), cover_rate = mean(fav_cover == "cover"), .groups = "drop")
cat("Cover rate by season:\n")
print(season_cover_rates, n = 20)

# ============================================================
# 2. Profile the failed 2025 cover-side bets
# ============================================================
cat("\n=== 2025 COVER-SIDE BETS (P > 0.54) ===\n")
bets_2025 <- p |> filter(test_season == 2025, p_cover > 0.54)
cat(sprintf("N = %d bets, cover rate = %.1f%%\n", nrow(bets_2025), mean(bets_2025$truth_cover) * 100))

cat("\nCompare to 2023 cover-side bets:\n")
bets_2023 <- p |> filter(test_season == 2023, p_cover > 0.54)
cat(sprintf("2023: N = %d, cover rate = %.1f%%\n", nrow(bets_2023), mean(bets_2023$truth_cover) * 100))
bets_2024 <- p |> filter(test_season == 2024, p_cover > 0.54)
cat(sprintf("2024: N = %d, cover rate = %.1f%%\n", nrow(bets_2024), mean(bets_2024$truth_cover) * 100))

# ============================================================
# 3. TRAINING WINDOW SWEEP
# ============================================================
cat("\n=== TRAINING WINDOW SWEEP ===\n")
cat("Testing different training start years for ALL test seasons\n\n")

run_model <- function(train_start, test_season, feats) {
  train <- v3 |> filter(season >= train_start, season < test_season)
  test <- v3 |> filter(season == test_season)
  if (nrow(train) < 100) return(NULL)

  train_x <- train |> select(all_of(feats))
  meds <- map_dbl(train_x, ~ median(.x, na.rm = TRUE))
  train_x <- train_x |> mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()
  test_x <- test |> select(all_of(feats)) |>
    mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()

  m <- colMeans(train_x); s <- apply(train_x, 2, sd); s[s == 0] <- 1
  train_xs <- scale(train_x, m, s); test_xs <- scale(test_x, m, s)
  gc <- which(apply(train_xs, 2, sd) > 0)
  if (length(gc) < 5) return(NULL)
  train_xs <- train_xs[, gc]; test_xs <- test_xs[, gc]

  y <- factor(train$fav_cover, levels = c("cover", "no_cover"))

  set.seed(301)
  cv_fit <- tryCatch(
    cv.glmnet(train_xs, y, family = "binomial", alpha = 0, nfolds = min(10, nrow(train) %/% 30),
               lambda = 10^seq(-4, -0.5, length.out = 40)),
    error = function(e) NULL
  )
  if (is.null(cv_fit)) return(NULL)

  pred <- predict(cv_fit, test_xs, s = "lambda.min", type = "response")[, 1]
  p_cover <- 1 - pred

  tibble(
    train_start = train_start, test_season = test_season,
    train_n = nrow(train), n_features = sum(coef(cv_fit, s = "lambda.min")[-1, 1] != 0),
    lambda = cv_fit$lambda.min,
    overall_acc = mean((p_cover > 0.5) == (test$fav_cover == "cover")),
    cover_bets = sum(p_cover > 0.54),
    cover_wr = if (sum(p_cover > 0.54) >= 5) mean((test$fav_cover == "cover")[p_cover > 0.54]) else NA_real_,
    cover_profit = if (sum(p_cover > 0.54) >= 5) sum(ifelse((test$fav_cover == "cover")[p_cover > 0.54], 100, -110)) else NA_real_
  )
}

# Test all combinations
window_results <- list()
for (start in c(2012, 2016, 2018, 2020, 2022, 2023)) {
  for (test_s in c(2023, 2024, 2025)) {
    if (start >= test_s) next
    res <- run_model(start, test_s, stable_feats)
    if (!is.null(res)) window_results[[length(window_results) + 1]] <- res
  }
}

wr <- bind_rows(window_results)

cat("Results by training start year:\n\n")
cat(sprintf("%-6s %-5s %5s %5s %6s %5s %6s %7s\n",
  "Start", "Test", "TrainN", "Feat", "AllAcc", "Bets", "CovWR", "Profit"))
cat(paste(rep("-", 55), collapse = ""), "\n")

for (i in seq_len(nrow(wr))) {
  r <- wr[i, ]
  cat(sprintf("%-6d %-5d %5d %5d %5.1f%% %5d %5.1f%% $%6.0f\n",
    r$train_start, r$test_season, r$train_n, r$n_features,
    r$overall_acc * 100, r$cover_bets,
    ifelse(is.na(r$cover_wr), 0, r$cover_wr * 100),
    ifelse(is.na(r$cover_profit), 0, r$cover_profit)))
}

# Summarize by start year across all test seasons
cat("\n=== SUMMARY BY TRAINING START ===\n")
wr |>
  group_by(train_start) |>
  summarize(
    seasons_tested = n(),
    mean_overall = round(mean(overall_acc) * 100, 1),
    total_cover_bets = sum(cover_bets, na.rm = TRUE),
    mean_cover_wr = round(mean(cover_wr, na.rm = TRUE) * 100, 1),
    total_cover_profit = sum(cover_profit, na.rm = TRUE),
    .groups = "drop"
  ) |> print()

# Best window for 2025 specifically
cat("\n=== 2025 BY TRAINING WINDOW ===\n")
wr |> filter(test_season == 2025) |>
  select(train_start, train_n, overall_acc, cover_bets, cover_wr, cover_profit) |>
  print()

write_csv(wr, here("results", "training_window_sweep.csv"))
cat("\nDone.\n")
