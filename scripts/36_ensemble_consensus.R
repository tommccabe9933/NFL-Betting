# Ensemble Consensus with Measured Independence
# Step 1: Measure prediction correlation between models
# Step 2: Build consensus strategies
# Step 3: Threshold sweep

library(tidyverse)
library(here)
library(glmnet)

source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

flip_features <- c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff","rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa","injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged","fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3","fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3","off_success_rate_roll5_diff","def_pass_epa_roll3_diff","fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff","fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5","fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5","dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff","off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff","def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
stable_feats <- setdiff(v3 |> select(where(is.numeric), -season, -week) |> names(), flip_features)

cat("=== ENSEMBLE CONSENSUS WITH MEASURED INDEPENDENCE ===\n\n")

# ============================================================
# 1. Generate predictions from 3 models
# ============================================================

# Helper function
run_ridge <- function(train_start, alpha, lambda) {
  all_preds <- list()
  for (test_s in c(2023, 2024, 2025)) {
    train <- v3 |> filter(season >= train_start, season < test_s)
    test <- v3 |> filter(season == test_s)
    if (nrow(train) < 50) next

    tc <- map_dbl(stable_feats, ~ mean(!is.na(train[[.x]])))
    feats <- stable_feats[tc > 0.50]

    train_x <- train |> select(all_of(feats))
    meds <- map_dbl(train_x, ~ median(.x, na.rm=TRUE))
    train_x <- train_x |> mutate(across(everything(), ~replace_na(.x, meds[cur_column()]))) |> as.matrix()
    test_x <- test |> select(all_of(feats)) |>
      mutate(across(everything(), ~replace_na(.x, meds[cur_column()]))) |> as.matrix()

    m <- colMeans(train_x); s <- apply(train_x,2,sd); s[s==0] <- 1
    train_xs <- scale(train_x,m,s); test_xs <- scale(test_x,m,s)
    gc <- which(apply(train_xs,2,sd) > 0)
    train_xs <- train_xs[,gc]; test_xs <- test_xs[,gc]
    y <- factor(train$fav_cover, levels=c("cover","no_cover"))

    fit <- glmnet(train_xs, y, family="binomial", alpha=alpha, lambda=lambda)
    pred <- 1 - predict(fit, test_xs, type="response")[,1]

    all_preds[[as.character(test_s)]] <- tibble(
      game_id_norm = test$game_id_norm, p_cover = pred, test_season = test_s
    )
  }
  bind_rows(all_preds)
}

cat("Generating predictions from 3 models...\n")

# Model 1: Best ridge (train=2022, a=0.1, lam=0.005)
ridge1 <- run_ridge(2022, 0.1, 0.005) |> rename(p_ridge1 = p_cover)
cat("  Ridge1 (train=2022): done\n")

# Model 2: Alt ridge (train=2016, a=0, lam=0.08)
ridge2 <- run_ridge(2016, 0, 0.08) |> rename(p_ridge2 = p_cover)
cat("  Ridge2 (train=2016): done\n")

# Model 3: XGBoost (load from Script 35)
xgb_path <- here("objects", "xgb_predictions.rds")
if (file.exists(xgb_path)) {
  xgb <- read_rds(xgb_path) |> select(game_id_norm, p_xgb = p_cover, test_season)
  cat("  XGBoost: loaded\n")
} else {
  cat("  XGBoost not yet available — running without it\n")
  xgb <- NULL
}

# ============================================================
# 2. Measure prediction correlation
# ============================================================
cat("\n=== PREDICTION CORRELATIONS ===\n")

combined <- ridge1 |>
  inner_join(ridge2, by = c("game_id_norm", "test_season"))

if (!is.null(xgb)) {
  combined <- combined |>
    inner_join(xgb, by = c("game_id_norm", "test_season"))
}

# Add truth
combined <- combined |>
  left_join(v3 |> select(game_id_norm, fav_cover), by = "game_id_norm") |>
  mutate(truth_cover = fav_cover == "cover")

cat(sprintf("  Ridge1 vs Ridge2: %.3f\n", cor(combined$p_ridge1, combined$p_ridge2)))
if (!is.null(xgb)) {
  cat(sprintf("  Ridge1 vs XGBoost: %.3f\n", cor(combined$p_ridge1, combined$p_xgb)))
  cat(sprintf("  Ridge2 vs XGBoost: %.3f\n", cor(combined$p_ridge2, combined$p_xgb)))
}

# Bet overlap at P > 0.55
r1_bets <- combined$p_ridge1 > 0.55
r2_bets <- combined$p_ridge2 > 0.55
overlap_r1r2 <- sum(r1_bets & r2_bets) / max(sum(r1_bets | r2_bets), 1)
cat(sprintf("\n  Bet overlap (P>0.55) Ridge1∩Ridge2: %.1f%% (%d games agree, %d total)\n",
  overlap_r1r2 * 100, sum(r1_bets & r2_bets), sum(r1_bets | r2_bets)))

if (!is.null(xgb)) {
  x_bets <- combined$p_xgb > 0.55
  overlap_r1x <- sum(r1_bets & x_bets) / max(sum(r1_bets | x_bets), 1)
  cat(sprintf("  Bet overlap (P>0.55) Ridge1∩XGB: %.1f%%\n", overlap_r1x * 100))
}

# ============================================================
# 3. Build ensemble strategies
# ============================================================
cat("\n=== ENSEMBLE STRATEGIES ===\n")

# Weighted average ensemble
combined <- combined |> mutate(
  p_ensemble_rr = 0.5 * p_ridge1 + 0.5 * p_ridge2
)
if (!is.null(xgb)) {
  combined <- combined |> mutate(
    p_ensemble_all = 0.4 * p_ridge1 + 0.3 * p_ridge2 + 0.3 * p_xgb,
    p_ensemble_rx = 0.5 * p_ridge1 + 0.5 * p_xgb,
    # Consensus: count how many models say P > 0.54
    n_agree_054 = (p_ridge1 > 0.54) + (p_ridge2 > 0.54) + (p_xgb > 0.54),
    n_agree_052 = (p_ridge1 > 0.52) + (p_ridge2 > 0.52) + (p_xgb > 0.52)
  )
}

# ============================================================
# 4. Threshold sweep for all ensemble methods
# ============================================================
cat("\n=== THRESHOLD SWEEP: ALL METHODS (cover-side only) ===\n\n")

methods <- c("p_ridge1", "p_ridge2", "p_ensemble_rr")
if (!is.null(xgb)) {
  methods <- c(methods, "p_xgb", "p_ensemble_all", "p_ensemble_rx")
}

for (method in methods) {
  cat(sprintf("--- %s ---\n", method))
  for (th in seq(0.52, 0.60, by = 0.01)) {
    bets <- combined |> filter(.data[[method]] > th)
    if (nrow(bets) < 10) next
    wr <- mean(bets$truth_cover)
    profit <- sum(ifelse(bets$truth_cover, 100, -110))
    cat(sprintf("  P>%.2f: %4d bets | win: %5.1f%% | $%6.0f | ROI: %5.1f%%\n",
      th, nrow(bets), wr * 100, profit, profit / (nrow(bets) * 110) * 100))
  }
  cat("\n")
}

# Consensus strategies
if (!is.null(xgb)) {
  cat("--- CONSENSUS: 2-of-3 agree P > 0.54 ---\n")
  bets_2of3 <- combined |> filter(n_agree_054 >= 2)
  if (nrow(bets_2of3) >= 5) {
    wr <- mean(bets_2of3$truth_cover)
    profit <- sum(ifelse(bets_2of3$truth_cover, 100, -110))
    cat(sprintf("  %d bets | win: %.1f%% | $%.0f\n\n", nrow(bets_2of3), wr * 100, profit))
  }

  cat("--- CONSENSUS: 3-of-3 agree P > 0.52 ---\n")
  bets_3of3 <- combined |> filter(n_agree_052 >= 3)
  if (nrow(bets_3of3) >= 5) {
    wr <- mean(bets_3of3$truth_cover)
    profit <- sum(ifelse(bets_3of3$truth_cover, 100, -110))
    cat(sprintf("  %d bets | win: %.1f%% | $%.0f\n\n", nrow(bets_3of3), wr * 100, profit))
  }
}

write_rds(combined, here("objects", "ensemble_predictions.rds"))
cat("Done.\n")
