# Shared model code for the v3.5 cover / no-cover ensemble
# Used by 44_bootstrap_combined.R (walk-forward backtest) and 47_weekly_picks.R.

library(tidyverse); library(glmnet)

# Features whose sign flips between eras, and engineered features that hurt
flip_features <- c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff",
  "rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa",
  "injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged",
  "fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3",
  "fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3",
  "off_success_rate_roll5_diff","def_pass_epa_roll3_diff",
  "fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff",
  "fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5",
  "fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5",
  "dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff",
  "off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff",
  "def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
bad_new <- c("total_spread_script","high_total_big_spread","dog_cold_streak",
  "dog_hot_streak","fade_dog_heat","qb_scheme_fit","defensive_style_clash",
  "fav_expected_score","dog_expected_score","fav_pass_lean","dog_def_weakness")

stable_features <- function(data) {
  setdiff(data |> select(where(is.numeric), -season, -week) |> names(),
          c(flip_features, bad_new))
}

# Ensemble members: training start season, glmnet alpha, lambda
cover_members <- list(A = c(2022, 0, 0.10), B = c(2022, 0, 0.005), C = c(2022, 0.5, 0.005))
nocover_member <- c(2012, 0.25, 0.08)
cover_threshold <- 0.55
nocover_threshold <- 0.53

# Fit on `tr`, predict `te`. Returns P(cover) or P(no_cover).
fit_predict <- function(tr, te, sf, alpha_val, lambda_val, target_cover = TRUE) {
  tc <- map_dbl(sf, ~ mean(!is.na(tr[[.x]])))
  fe <- sf[tc > 0.5]
  tx <- tr |> select(all_of(fe))
  md <- map_dbl(tx, ~ median(.x, na.rm = TRUE))
  tx <- tx |> mutate(across(everything(), ~ replace_na(.x, md[cur_column()]))) |> as.matrix()
  ex <- te |> select(all_of(fe)) |>
    mutate(across(everything(), ~ replace_na(.x, md[cur_column()]))) |> as.matrix()

  m <- colMeans(tx); s <- apply(tx, 2, sd); s[s == 0] <- 1
  txs <- scale(tx, m, s); exs <- scale(ex, m, s)
  gc <- which(apply(txs, 2, sd) > 0)
  txs <- txs[, gc]; exs <- exs[, gc, drop = FALSE]

  lv <- if (target_cover) c("cover", "no_cover") else c("no_cover", "cover")
  y <- factor(tr$fav_cover, levels = lv)
  fit <- glmnet(txs, y, family = "binomial", alpha = alpha_val, lambda = lambda_val)
  1 - predict(fit, exs, type = "response")[, 1]
}
