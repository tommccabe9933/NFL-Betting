# Brier Decomposition & Error Analysis
# Determines: is failure calibration or discrimination?
# Segments errors by spread bucket, division, weather, timing, etc.

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "14_research_protocol.R"))

# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================
if (file.exists(here("objects", "temporal_cv_predictions.rds"))) {
  preds <- read_rds(here("objects", "temporal_cv_predictions.rds"))
  cat("Using temporal CV predictions:", nrow(preds), "games\n")
} else {
  stop("No predictions found — run script 15 first")
}

prob_col <- ".pred_cover_calibrated"
if (!prob_col %in% names(preds)) {
  prob_col <- ".pred_cover"
  cat("WARNING: Using uncalibrated predictions\n")
}

preds <- preds |>
  mutate(
    truth_num = as.integer(fav_cover == "cover"),
    pred_p = .data[[prob_col]],
    residual = truth_num - pred_p,
    abs_error = abs(residual),
    squared_error = residual^2
  )

# ============================================================
# 2. BRIER DECOMPOSITION
# ============================================================
cat("\n=== BRIER DECOMPOSITION ===\n")

# Murphy decomposition: Brier = Reliability - Resolution + Uncertainty
# Bin predictions into 10 bins
n_bins <- 10
preds_binned <- preds |>
  mutate(
    pred_bin = cut(pred_p, breaks = seq(0, 1, length.out = n_bins + 1),
                   include.lowest = TRUE)
  )

bin_stats <- preds_binned |>
  group_by(pred_bin) |>
  summarize(
    n_k = n(),
    mean_pred = mean(pred_p),
    mean_obs = mean(truth_num),
    .groups = "drop"
  )

N <- nrow(preds)
overall_mean <- mean(preds$truth_num)

# Reliability (calibration error): weighted MSE of bin means vs observed
reliability <- sum(bin_stats$n_k * (bin_stats$mean_pred - bin_stats$mean_obs)^2) / N

# Resolution (discrimination power): weighted variance of bin observed rates
resolution <- sum(bin_stats$n_k * (bin_stats$mean_obs - overall_mean)^2) / N

# Uncertainty (irreducible): overall_mean * (1 - overall_mean)
uncertainty <- overall_mean * (1 - overall_mean)

brier_total <- mean(preds$squared_error)

cat(sprintf("  Brier Score:   %.5f\n", brier_total))
cat(sprintf("  Reliability:   %.5f (calibration error — lower is better)\n", reliability))
cat(sprintf("  Resolution:    %.5f (discrimination — higher is better)\n", resolution))
cat(sprintf("  Uncertainty:   %.5f (irreducible)\n", uncertainty))
cat(sprintf("  Check: R - Res + Unc = %.5f\n", reliability - resolution + uncertainty))
cat(sprintf("  Coin flip Brier: %.5f\n", 0.25))

# Diagnosis
if (reliability > resolution) {
  cat("\n  DIAGNOSIS: Calibration error > discrimination power.\n")
  cat("  The model is MORE miscalibrated than it is informative.\n")
  cat("  Priority: improve calibration (isotonic regression, Venn-Abers)\n")
} else {
  cat("\n  DIAGNOSIS: Discrimination > calibration error.\n")
  cat("  The model has SOME discrimination but may still be poorly calibrated.\n")
  cat("  Priority: maintain calibration, improve discrimination (better features)\n")
}

decomp <- tibble(
  component = c("brier_total", "reliability", "resolution", "uncertainty"),
  value = c(brier_total, reliability, resolution, uncertainty)
)
write_csv(decomp, here("results", "brier_decomposition.csv"))

# ============================================================
# 3. ERROR SEGMENTATION
# ============================================================
cat("\n=== ERROR SEGMENTATION ===\n")
cat("Analyzing WHERE the model fails...\n\n")

# Load v2/v3 data for segmentation features
if (file.exists(here("objects", "modeling_data_v3.rds"))) {
  model_data <- read_rds(here("objects", "modeling_data_v3.rds"))
} else {
  model_data <- read_rds(here("objects", "modeling_data_v2.rds"))
}

# Merge segmentation features onto predictions
preds_full <- preds |>
  left_join(
    model_data |> select(game_id_norm, season, week, spread_fav, over_under_line,
                          fav_at_home, any_of(c("div_game", "is_primetime", "is_dome",
                                                 "weather_wind_mph", "weather_temperature",
                                                 "fav_short_week", "dog_short_week",
                                                 "rest_diff", "fav_qb_out", "dog_qb_out"))),
    by = "game_id_norm"
  )

# Create segments
preds_seg <- preds_full |>
  mutate(
    abs_spread = abs(spread_fav),
    spread_bucket = case_when(
      abs_spread <= 3   ~ "1-3",
      abs_spread <= 6.5 ~ "3.5-6.5",
      abs_spread <= 9.5 ~ "7-9.5",
      abs_spread <= 13.5 ~ "10-13.5",
      TRUE               ~ "14+"
    ),
    season_phase = case_when(
      week.x <= 4  ~ "early (1-4)",
      week.x <= 12 ~ "mid (5-12)",
      TRUE          ~ "late (13-18)"
    ),
    home_fav = if_else(fav_at_home == "yes", "home_fav", "away_fav"),
    wind_bucket = case_when(
      is.na(weather_wind_mph) ~ "dome/NA",
      weather_wind_mph < 10   ~ "calm (<10)",
      weather_wind_mph < 20   ~ "windy (10-20)",
      TRUE                     ~ "extreme (20+)"
    ),
    qb_change = if_else(
      (fav_qb_out == TRUE) | (dog_qb_out == TRUE),
      "qb_change", "stable", missing = "stable"
    )
  )

# Wilson confidence interval helper
wilson_ci <- function(n_success, n_total, confidence = 0.95) {
  if (n_total == 0) return(c(NA, NA, NA))
  p <- n_success / n_total
  z <- qnorm((1 + confidence) / 2)
  denom <- 1 + z^2 / n_total
  center <- (p + z^2 / (2 * n_total)) / denom
  margin <- z * sqrt(p * (1 - p) / n_total + z^2 / (4 * n_total^2)) / denom
  c(lower = center - margin, estimate = p, upper = center + margin)
}

# Compute accuracy by segment
segment_vars <- c("spread_bucket", "season_phase", "home_fav", "wind_bucket",
                   "qb_change")
if ("div_game" %in% names(preds_seg)) segment_vars <- c(segment_vars, "div_game")
if ("is_primetime" %in% names(preds_seg)) segment_vars <- c(segment_vars, "is_primetime")

segments_list <- list()

for (seg_var in segment_vars) {
  if (!seg_var %in% names(preds_seg)) next

  seg_result <- preds_seg |>
    mutate(.seg_val = as.character(.data[[seg_var]])) |>
    group_by(segment_value = .seg_val) |>
    summarize(
      n = n(),
      n_correct = sum((pred_p > 0.5) == (truth_num == 1)),
      accuracy = mean((pred_p > 0.5) == (truth_num == 1)),
      mean_brier = mean(squared_error),
      mean_log_loss = -mean(truth_num * log(pmax(pred_p, 1e-10)) +
                             (1 - truth_num) * log(pmax(1 - pred_p, 1e-10))),
      mean_pred = mean(pred_p),
      mean_obs = mean(truth_num),
      .groups = "drop"
    ) |>
    mutate(
      segment_var = seg_var,
      ci_lower = map2_dbl(n_correct, n, ~ wilson_ci(.x, .y)[1]),
      ci_upper = map2_dbl(n_correct, n, ~ wilson_ci(.x, .y)[3]),
      # Is this segment significantly above 52.4% (breakeven)?
      above_breakeven = ci_lower > 0.524,
      # Is this segment significantly above 50%?
      above_coinflip = ci_lower > 0.50
    )

  segments_list[[seg_var]] <- seg_result
}

segments <- bind_rows(segments_list)

cat("Segment results:\n")
segments |>
  arrange(desc(accuracy)) |>
  select(segment_var, segment_value, n, accuracy, ci_lower, ci_upper,
         mean_brier, above_breakeven) |>
  print(n = 40)

# Flag promising segments
promising <- segments |> filter(above_coinflip | accuracy > 0.54)
if (nrow(promising) > 0) {
  cat("\n=== PROMISING SEGMENTS (accuracy > 54% or CI above 50%) ===\n")
  print(promising |> select(segment_var, segment_value, n, accuracy, ci_lower, ci_upper))
} else {
  cat("\nNo segments show statistically significant accuracy above 50%\n")
}

write_csv(segments, here("results", "error_segments.csv"))

# ============================================================
# 4. RESIDUAL CORRELATION ANALYSIS
# ============================================================
cat("\n=== RESIDUAL CORRELATION WITH FEATURES ===\n")
cat("Do residuals correlate with any feature? (If yes = unexploited signal)\n\n")

numeric_cols <- model_data |>
  select(where(is.numeric)) |>
  select(-any_of(c("season", "week"))) |>
  names()

resid_data <- preds |>
  select(game_id_norm, residual) |>
  left_join(model_data |> select(game_id_norm, all_of(numeric_cols)), by = "game_id_norm")

correlations <- map_dbl(numeric_cols, function(col) {
  x <- resid_data[[col]]
  y <- resid_data$residual
  valid <- !is.na(x) & !is.na(y)
  if (sum(valid) < 30) return(NA_real_)
  cor(x[valid], y[valid])
})

cor_df <- tibble(feature = numeric_cols, residual_cor = correlations) |>
  filter(!is.na(residual_cor)) |>
  arrange(desc(abs(residual_cor)))

cat("Top 15 features correlated with prediction residuals:\n")
print(cor_df |> head(15))

# Any strong residual correlations suggest unexploited signal
strong <- cor_df |> filter(abs(residual_cor) > 0.05)
if (nrow(strong) > 0) {
  cat("\n=== FEATURES WITH |residual_cor| > 0.05 (potential unexploited signal) ===\n")
  print(strong)
} else {
  cat("\nNo features with strong residual correlation — model is using available signal\n")
}

write_csv(cor_df, here("results", "residual_correlations.csv"))

# ============================================================
# 5. CALIBRATION CURVE
# ============================================================
cat("\n=== CALIBRATION CURVE ===\n")
cal_curve <- preds |>
  mutate(pred_bin = cut(pred_p, breaks = seq(0.3, 0.7, by = 0.02), include.lowest = TRUE)) |>
  group_by(pred_bin) |>
  summarize(
    n = n(),
    mean_pred = mean(pred_p),
    mean_obs = mean(truth_num),
    se = sqrt(mean_obs * (1 - mean_obs) / pmax(n, 1)),
    .groups = "drop"
  ) |>
  filter(n >= 10)

print(cal_curve)

cat("\nDone! Results saved to results/brier_decomposition.csv, error_segments.csv, residual_correlations.csv\n")
