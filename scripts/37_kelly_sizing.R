# Kelly Criterion Sizing — Calibration-First
# SUPERSEDED by 45_calibrated_kelly.R (v3.5). Kept for the record; its inputs
# (modeling_data_v3.rds, tuned_stable_predictions.rds) now live in archive/.
# Step 0: Calibration audit in betting region
# Step 1: Kelly implementation (only if calibration passes)
# Step 2: Bankroll simulation

library(tidyverse)
library(here)

cat("=== KELLY SIZING WITH CALIBRATION AUDIT ===\n\n")

# Load best strategy predictions
# Use our current best: train=2022, a=0.1, lam=0.005, 3-season window
# Need to regenerate with game ordering for chronological Kelly

v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

# Load tuned predictions
p <- read_rds(here("objects", "tuned_stable_predictions.rds"))

# ============================================================
# STEP 0: Calibration audit in the BETTING REGION
# ============================================================
cat("=== CALIBRATION AUDIT (betting region only) ===\n")
cat("For games where P(cover) > 0.52 (our bet universe)\n\n")

betting_region <- p |> filter(p_cover > 0.52)
cat(sprintf("Betting region: %d games\n", nrow(betting_region)))

# Bin into deciles by P(cover)
betting_region <- betting_region |>
  mutate(p_bucket = cut(p_cover, breaks = quantile(p_cover, seq(0, 1, 0.2)),
                         include.lowest = TRUE))

cal_audit <- betting_region |>
  group_by(p_bucket) |>
  summarize(
    n = n(),
    mean_predicted = mean(p_cover),
    actual_cover_rate = mean(truth_cover),
    brier = mean((p_cover - as.integer(truth_cover))^2),
    .groups = "drop"
  )

cat("Calibration in betting region (quintiles):\n")
cat(sprintf("%-20s %5s %8s %8s %8s\n", "Bucket", "N", "Predicted", "Actual", "Brier"))
cat(paste(rep("-", 55), collapse = ""), "\n")
for (i in seq_len(nrow(cal_audit))) {
  r <- cal_audit[i, ]
  cat(sprintf("%-20s %5d %7.3f   %7.3f   %.5f\n",
    as.character(r$p_bucket), r$n, r$mean_predicted, r$actual_cover_rate, r$brier))
}

# Calibration slope: regress actual on predicted
cal_slope <- lm(actual_cover_rate ~ mean_predicted, data = cal_audit)
slope <- coef(cal_slope)[2]
cat(sprintf("\nCalibration slope: %.3f (ideal = 1.0)\n", slope))

if (slope > 0 && slope > 0.3) {
  cat("PASS: Positive slope, higher predicted → higher actual. Kelly is safe.\n")
  kelly_safe <- TRUE
} else if (slope <= 0) {
  cat("FAIL: Negative slope — higher predicted does NOT mean higher actual.\n")
  cat("Kelly would be DANGEROUS. Stick with flat betting.\n")
  kelly_safe <- FALSE
} else {
  cat("MARGINAL: Positive but weak slope. Use conservative Kelly only.\n")
  kelly_safe <- TRUE
}

# Is predicted systematically over or under?
mean_pred <- mean(betting_region$p_cover)
mean_actual <- mean(betting_region$truth_cover)
cat(sprintf("\nMean predicted: %.3f | Mean actual: %.3f\n", mean_pred, mean_actual))
if (mean_actual > mean_pred) {
  cat("Model UNDERESTIMATES cover probability → Kelly will be conservative (OK)\n")
} else {
  cat("Model OVERESTIMATES cover probability → Kelly may oversize (DANGER)\n")
}

# ============================================================
# STEP 1: Kelly Implementation
# ============================================================
if (kelly_safe) {
  cat("\n=== KELLY SIZING ===\n\n")

  # Use the best ridge model: regenerate predictions with game ordering
  # For Kelly we need chronological order within each season

  betting_games <- p |>
    filter(p_cover > 0.55) |>  # Use our best threshold
    left_join(v3 |> select(game_id_norm, week), by = "game_id_norm") |>
    arrange(test_season, week)

  cat(sprintf("Betting games (P > 0.55): %d\n\n", nrow(betting_games)))

  # Simulate bankroll for each Kelly fraction
  for (kelly_mult in c(0.25, 0.5, 1.0)) {
    bankroll <- 10000
    peak <- bankroll
    trough <- bankroll
    max_drawdown <- 0
    history <- tibble(bet_num = 0, bankroll = bankroll, season = NA_real_)

    for (i in seq_len(nrow(betting_games))) {
      game <- betting_games[i, ]
      p_win <- game$p_cover
      edge <- p_win - 0.524  # breakeven at -110

      if (edge <= 0) next  # skip negative edge

      # Kelly fraction
      kelly_f <- edge * 2.1  # full Kelly at -110: (b*p - q)/b = 2.1 * (p - 0.524)
      bet_size <- kelly_f * bankroll * kelly_mult
      bet_size <- min(bet_size, 0.05 * bankroll)  # cap at 5%
      bet_size <- max(bet_size, 10)  # minimum bet

      # Outcome
      if (game$truth_cover) {
        bankroll <- bankroll + bet_size * (100 / 110)  # win pays 100/110
      } else {
        bankroll <- bankroll - bet_size  # lose the bet
      }

      peak <- max(peak, bankroll)
      trough <- min(trough, bankroll)
      max_drawdown <- max(max_drawdown, (peak - bankroll) / peak)
      history <- bind_rows(history, tibble(bet_num = i, bankroll = bankroll,
                                            season = game$test_season))
    }

    total_return <- (bankroll - 10000) / 10000

    cat(sprintf("Kelly %.0f%%: Final=$%.0f | Return=%.1f%% | Peak=$%.0f | Trough=$%.0f | MaxDD=%.1f%%\n",
      kelly_mult * 100, bankroll, total_return * 100, peak, trough, max_drawdown * 100))
  }

  # Flat betting comparison
  flat_wins <- sum(betting_games$truth_cover)
  flat_losses <- nrow(betting_games) - flat_wins
  flat_profit <- flat_wins * 100 - flat_losses * 110
  cat(sprintf("\nFlat $100: Profit=$%.0f | ROI=%.1f%% | Win rate=%.1f%%\n",
    flat_profit, flat_profit / (nrow(betting_games) * 110) * 100,
    flat_wins / nrow(betting_games) * 100))

} else {
  cat("\nKelly SKIPPED — calibration failed. Flat betting recommended.\n")
}

cat("\nDone.\n")
