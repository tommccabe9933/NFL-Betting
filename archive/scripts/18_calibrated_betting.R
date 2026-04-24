# Calibrated Betting Strategy
# Uses calibrated probabilities from walk-forward pipeline
# Bets only when lower confidence bound on EV > 0
# Reports uncertainty bands, not just point estimates

library(tidyverse)
library(tidymodels)
library(here)
library(probably)

source(here("scripts", "14_research_protocol.R"))

# Load predictions from temporal CV or base+overlay
# Try base model first, fall back to temporal CV
if (file.exists(here("objects", "base_model_predictions.rds"))) {
  preds <- read_rds(here("objects", "base_model_predictions.rds"))
  cat("Using base model predictions\n")
  prob_col <- ".pred_cover_calibrated"
} else if (file.exists(here("objects", "temporal_cv_predictions.rds"))) {
  preds <- read_rds(here("objects", "temporal_cv_predictions.rds"))
  cat("Using temporal CV predictions\n")
  prob_col <- ".pred_cover_calibrated"
} else {
  # Fall back to expanding advanced predictions (uncalibrated)
  preds <- read_rds(here("objects", "expanding_advanced_predictions.rds"))
  cat("WARNING: Using uncalibrated predictions (run scripts 15/16 first)\n")
  prob_col <- ".pred_cover"
  preds$.pred_cover_calibrated <- preds$.pred_cover
}

cat("Predictions:", nrow(preds), "games\n\n")

# ============================================================
# 1. CALIBRATION DIAGNOSTIC
# ============================================================
cat("=== CALIBRATION DIAGNOSTIC ===\n")
cal_data <- calibration_curve_data(preds, prob_col = prob_col, n_bins = 10)
cat("Calibration curve (predicted vs observed):\n")
print(cal_data)

# Overall calibration metrics
overall_brier <- brier_score(preds$fav_cover, preds[[prob_col]], "first")
overall_ll <- log_loss(preds$fav_cover, preds[[prob_col]], "first")
cat("\nBrier score:", round(overall_brier, 4), "(coin flip = 0.2500)\n")
cat("Log loss:", round(overall_ll, 4), "(coin flip = 0.6931)\n")

# ============================================================
# 2. EV-BASED BETTING (not threshold-based)
# ============================================================
cat("\n=== EV-BASED BETTING ===\n")
cat("Bet only when expected value is positive given uncertainty\n\n")

preds_ev <- preds |>
  mutate(
    cal_p = .data[[prob_col]],
    # For cover bet: p is probability of cover
    ev_cover = compute_ev(cal_p),
    # For no-cover bet: p is probability of no-cover = 1 - cal_p
    ev_nocover = compute_ev(1 - cal_p),
    # Best EV side
    best_ev = pmax(ev_cover, ev_nocover),
    best_side = if_else(ev_cover >= ev_nocover, "cover", "no_cover"),
    # Actual outcome
    actual = as.character(fav_cover),
    # Would the bet win?
    bet_correct = best_side == actual,
    profit = if_else(bet_correct, 100, -110)
  )

# EV threshold sweep (with uncertainty)
cat("EV threshold analysis:\n")
cat("(Only bet when point EV > threshold)\n\n")

for (ev_thresh in c(0, 1, 2, 3, 4, 5, 7, 10)) {
  bets <- preds_ev |> filter(best_ev > ev_thresh)
  if (nrow(bets) < 20) next

  win_rate <- mean(bets$bet_correct)
  total_profit <- sum(bets$profit)
  roi <- total_profit / (nrow(bets) * 110) * 100

  # Uncertainty on win rate
  se <- sqrt(win_rate * (1 - win_rate) / nrow(bets))
  ci_low <- win_rate - 1.645 * se  # 90% CI
  ci_high <- win_rate + 1.645 * se

  cat(sprintf("EV > $%2d: %4d bets (%4.1f%%) | win: %.1f%% [%.1f%%-%.1f%%] | profit: $%6.0f | ROI: %5.1f%%\n",
              ev_thresh, nrow(bets), nrow(bets)/nrow(preds_ev)*100,
              win_rate*100, ci_low*100, ci_high*100,
              total_profit, roi))
}

# ============================================================
# 3. CONSERVATIVE EV: Lower confidence bound
# ============================================================
cat("\n=== CONSERVATIVE: Lower bound EV ===\n")
cat("Bet only when 80% lower bound on EV > 0\n\n")

# Group predictions into probability bins and compute empirical win rates
preds_ev <- preds_ev |>
  mutate(prob_bin = round(cal_p * 20) / 20)  # bin to nearest 0.05

bin_stats <- preds_ev |>
  group_by(prob_bin) |>
  summarize(
    n = n(),
    empirical_p = mean(fav_cover == "cover"),
    .groups = "drop"
  )

# Join bin stats back
preds_ev <- preds_ev |>
  left_join(bin_stats |> select(prob_bin, n_similar = n), by = "prob_bin")

preds_ev <- preds_ev |>
  mutate(
    ev_lower_cover = compute_ev_lower_bound(cal_p, n_similar, confidence = 0.80),
    ev_lower_nocover = compute_ev_lower_bound(1 - cal_p, n_similar, confidence = 0.80),
    best_ev_lower = pmax(ev_lower_cover, ev_lower_nocover),
    conservative_bet = best_ev_lower > 0
  )

conservative_bets <- preds_ev |> filter(conservative_bet)

if (nrow(conservative_bets) > 0) {
  cat("Conservative bets:", nrow(conservative_bets), "of", nrow(preds_ev), "games\n")
  cat("Win rate:", round(mean(conservative_bets$bet_correct) * 100, 1), "%\n")
  cat("Total profit:", scales::dollar(sum(conservative_bets$profit)), "\n")
  cat("ROI:", round(sum(conservative_bets$profit) / (nrow(conservative_bets) * 110) * 100, 1), "%\n")

  # Season breakdown
  cat("\nSeason breakdown:\n")
  conservative_bets |>
    group_by(season = if ("test_season" %in% names(.)) test_season else season) |>
    summarize(
      n = n(), win_rate = round(mean(bet_correct) * 100, 1),
      profit = sum(profit),
      roi = round(sum(profit) / (n() * 110) * 100, 1),
      .groups = "drop"
    ) |> print()
} else {
  cat("No bets pass conservative threshold — model not confident enough\n")
  cat("This is an HONEST result: the model may not have a real edge\n")
}

# ============================================================
# 4. FLAT STAKING BANKROLL PATH
# ============================================================
cat("\n=== BANKROLL SIMULATION (flat $100 stake) ===\n")

# Use EV > $2 threshold as primary strategy
strategy_bets <- preds_ev |>
  filter(best_ev > 2) |>
  arrange(season, week) |>
  mutate(
    cumulative_profit = cumsum(profit),
    bankroll = 10000 + cumulative_profit,
    bet_number = row_number()
  )

if (nrow(strategy_bets) > 0) {
  cat("Starting bankroll: $10,000\n")
  cat("Total bets:", nrow(strategy_bets), "\n")
  cat("Final bankroll:", scales::dollar(tail(strategy_bets$bankroll, 1)), "\n")
  cat("Peak bankroll:", scales::dollar(max(strategy_bets$bankroll)), "\n")
  cat("Trough bankroll:", scales::dollar(min(strategy_bets$bankroll)), "\n")
  cat("Max drawdown:", scales::dollar(10000 - min(strategy_bets$bankroll)), "\n")
}

# ============================================================
# 5. SAVE RESULTS
# ============================================================

# Calibration plot
cal_plot <- ggplot(cal_data, aes(x = mean_predicted, y = mean_observed)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(size = n), color = "#013369") +
  geom_errorbar(aes(ymin = mean_observed - 1.96 * se,
                     ymax = mean_observed + 1.96 * se),
                width = 0.01, color = "#013369", alpha = 0.5) +
  scale_x_continuous(limits = c(0.3, 0.7)) +
  scale_y_continuous(limits = c(0.3, 0.7)) +
  labs(
    title = "Probability Calibration",
    subtitle = "Perfect calibration follows the diagonal line",
    x = "Predicted Probability (cover)",
    y = "Observed Frequency (cover)",
    size = "Games"
  ) +
  theme_minimal(base_size = 12)

dir.create(here("figures"), showWarnings = FALSE)
ggsave(here("figures", "calibration_curve.png"), cal_plot, width = 7, height = 6, dpi = 300)

# Save betting results
write_csv(
  preds_ev |> select(game_id_norm, season, week, fav_cover, cal_p,
                      ev_cover, ev_nocover, best_ev, best_side,
                      bet_correct, profit),
  here("results", "calibrated_betting_detail.csv")
)

cat("\nDone! Figures: figures/calibration_curve.png\n")
