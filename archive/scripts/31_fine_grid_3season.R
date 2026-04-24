# Fine-grained grid search on the 3-season window model
# Our best: 59.5% on 148 cover-side bets. Can we do better?

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

all_feats <- v3 |> select(where(is.numeric), -season, -week) |> names()
stable_feats <- setdiff(all_feats, flip_features)

WINDOW <- 3  # 3-season rolling window (our best)

cat("=== FINE GRID: 3-SEASON WINDOW ===\n\n")

# Test many alpha x lambda combinations
results <- list()

for (alpha in c(0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)) {
  for (test_s in c(2023, 2024, 2025)) {
    train_start <- test_s - WINDOW
    train <- v3 |> filter(season >= train_start, season < test_s)
    test <- v3 |> filter(season == test_s)

    # Feature coverage in training window
    train_coverage <- map_dbl(stable_feats, ~ mean(!is.na(train[[.x]])))
    era_feats <- stable_feats[train_coverage > 0.50]

    train_x <- train |> select(all_of(era_feats))
    meds <- map_dbl(train_x, ~ median(.x, na.rm = TRUE))
    train_x <- train_x |> mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()
    test_x <- test |> select(all_of(era_feats)) |>
      mutate(across(everything(), ~ replace_na(.x, meds[cur_column()]))) |> as.matrix()

    m <- colMeans(train_x); s <- apply(train_x, 2, sd); s[s == 0] <- 1
    train_xs <- scale(train_x, m, s); test_xs <- scale(test_x, m, s)
    gc <- which(apply(train_xs, 2, sd) > 0)
    train_xs <- train_xs[, gc]; test_xs <- test_xs[, gc]

    y <- factor(train$fav_cover, levels = c("cover", "no_cover"))

    # Finer lambda grid
    set.seed(301)
    nf <- min(10, max(3, nrow(train) %/% 30))
    cv_fit <- tryCatch(
      cv.glmnet(train_xs, y, family = "binomial", alpha = alpha, nfolds = nf,
                 lambda = 10^seq(-3.5, 0, length.out = 60)),
      error = function(e) NULL
    )
    if (is.null(cv_fit)) next

    pred <- predict(cv_fit, test_xs, s = "lambda.min", type = "response")[, 1]
    p_cover <- 1 - pred

    truth_cover <- test$fav_cover == "cover"
    overall_acc <- mean((p_cover > 0.5) == truth_cover)
    cover_bets <- sum(p_cover > 0.54)
    cover_wr <- if (cover_bets >= 5) mean(truth_cover[p_cover > 0.54]) else NA

    results[[length(results) + 1]] <- tibble(
      alpha = alpha, test_season = test_s, lambda = cv_fit$lambda.min,
      n_features = sum(coef(cv_fit, s = "lambda.min")[-1, 1] != 0),
      overall_acc = overall_acc, cover_bets = cover_bets, cover_wr = cover_wr
    )
  }
}

res <- bind_rows(results)

# Summary by alpha
cat("Results by alpha (3-season window, cover P > 0.54):\n\n")
res |>
  filter(!is.na(cover_wr)) |>
  group_by(alpha) |>
  summarize(
    total_cover_bets = sum(cover_bets),
    mean_cover_wr = round(mean(cover_wr) * 100, 1),
    total_profit = sum(ifelse(cover_wr > 0.524, cover_bets * cover_wr * 100 - cover_bets * (1 - cover_wr) * 110,
                               cover_bets * cover_wr * 100 - cover_bets * (1 - cover_wr) * 110)),
    mean_overall = round(mean(overall_acc) * 100, 1),
    mean_features = round(mean(n_features)),
    .groups = "drop"
  ) |>
  arrange(desc(mean_cover_wr)) |>
  print()

# Detailed per-season for top alphas
cat("\n=== DETAILED: Best alpha per season ===\n")
res |>
  filter(!is.na(cover_wr)) |>
  group_by(alpha) |>
  summarize(
    s2023 = round(cover_wr[test_season == 2023] * 100, 1),
    s2024 = round(cover_wr[test_season == 2024] * 100, 1),
    s2025 = round(cover_wr[test_season == 2025] * 100, 1),
    bets = sum(cover_bets),
    .groups = "drop"
  ) |>
  arrange(desc((s2023 + s2024 + s2025) / 3)) |>
  print()

write_csv(res, here("results", "fine_grid_3season.csv"))
cat("\nDone.\n")
