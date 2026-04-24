# Bootstrap Validation + Robustness Checks
# Is the 55.1% cover-side signal real?

library(tidyverse)
library(here)

p <- read_rds(here("objects", "tuned_stable_predictions.rds"))

cat("=== BOOTSTRAP VALIDATION ===\n\n")

# ============================================================
# 1. Bootstrap CI on cover-side P > 0.54
# ============================================================
cover_bets <- p |> filter(p_cover > 0.54)
n_bets <- nrow(cover_bets)
observed_wr <- mean(cover_bets$truth_cover)
observed_profit <- sum(ifelse(cover_bets$truth_cover, 100, -110))
observed_roi <- observed_profit / (n_bets * 110)

cat(sprintf("Cover-side P > 0.54: %d bets, %.1f%% win rate, $%.0f profit, %.1f%% ROI\n\n",
  n_bets, observed_wr * 100, observed_profit, observed_roi * 100))

set.seed(301)
B <- 5000
boot_wr <- replicate(B, {
  idx <- sample(n_bets, replace = TRUE)
  mean(cover_bets$truth_cover[idx])
})
boot_roi <- (boot_wr * 100 - (1 - boot_wr) * 110) / 110

cat("Bootstrap results (B=5000):\n")
cat(sprintf("  Win rate: median=%.1f%%, 90%% CI=[%.1f%%, %.1f%%]\n",
  median(boot_wr) * 100, quantile(boot_wr, 0.05) * 100, quantile(boot_wr, 0.95) * 100))
cat(sprintf("  ROI: median=%.1f%%, 90%% CI=[%.1f%%, %.1f%%]\n",
  median(boot_roi) * 100, quantile(boot_roi, 0.05) * 100, quantile(boot_roi, 0.95) * 100))
cat(sprintf("  P(win_rate > 52.4%%): %.1f%%\n", mean(boot_wr > 0.524) * 100))
cat(sprintf("  P(ROI > 0%%): %.1f%%\n", mean(boot_roi > 0) * 100))

# ============================================================
# 2. Threshold stability
# ============================================================
cat("\n=== THRESHOLD STABILITY ===\n")
cat("Is the curve monotonically increasing?\n\n")

thresh_curve <- map_dfr(seq(0.510, 0.600, by = 0.005), function(th) {
  bets <- p |> filter(p_cover > th)
  if (nrow(bets) < 5) return(NULL)
  tibble(
    threshold = th,
    n_bets = nrow(bets),
    win_rate = mean(bets$truth_cover),
    profit = sum(ifelse(bets$truth_cover, 100, -110))
  )
})

cat(sprintf("%-10s %6s %8s %8s\n", "Threshold", "Bets", "WinRate", "Profit"))
for (i in seq_len(nrow(thresh_curve))) {
  r <- thresh_curve[i, ]
  cat(sprintf("P > %.3f  %5d   %5.1f%%  $%6.0f\n",
    r$threshold, r$n_bets, r$win_rate * 100, r$profit))
}

# Check monotonicity
diffs <- diff(thresh_curve$win_rate)
n_increasing <- sum(diffs > 0)
n_decreasing <- sum(diffs < 0)
cat(sprintf("\nMonotonicity: %d increasing, %d decreasing steps\n", n_increasing, n_decreasing))
if (n_increasing > n_decreasing * 2) {
  cat("PASS: Generally monotonically increasing — signal is real\n")
} else {
  cat("MIXED: Not clearly monotonic — could be noise\n")
}

# ============================================================
# 3. Permutation test
# ============================================================
cat("\n=== PERMUTATION TEST ===\n")

set.seed(301)
n_perm <- 1000
perm_wrs <- replicate(n_perm, {
  shuffled <- p
  shuffled$truth_cover <- sample(shuffled$truth_cover)
  bets <- shuffled |> filter(p_cover > 0.54)
  mean(bets$truth_cover)
})

perm_pvalue <- mean(perm_wrs >= observed_wr)
cat(sprintf("Observed win rate: %.3f\n", observed_wr))
cat(sprintf("Permutation mean: %.3f, SD: %.3f\n", mean(perm_wrs), sd(perm_wrs)))
cat(sprintf("Permutation p-value: %.4f\n", perm_pvalue))
if (perm_pvalue < 0.05) {
  cat("SIGNIFICANT at 5%\n")
} else if (perm_pvalue < 0.10) {
  cat("SIGNIFICANT at 10%\n")
} else {
  cat("NOT significant\n")
}

# ============================================================
# 4. Leave-one-season-out
# ============================================================
cat("\n=== LEAVE-ONE-SEASON-OUT ===\n")
for (leave_out in c(2023, 2024, 2025)) {
  subset <- p |> filter(test_season != leave_out, p_cover > 0.54)
  if (nrow(subset) < 5) next
  wr <- mean(subset$truth_cover)
  profit <- sum(ifelse(subset$truth_cover, 100, -110))
  cat(sprintf("  Without %d: %d bets, %.1f%%, $%.0f\n", leave_out, nrow(subset), wr * 100, profit))
}

# Save
results <- tibble(
  metric = c("observed_wr", "observed_roi", "boot_median_wr", "boot_ci_lower", "boot_ci_upper",
             "p_above_breakeven", "p_roi_positive", "perm_pvalue"),
  value = c(observed_wr, observed_roi, median(boot_wr),
            quantile(boot_wr, 0.05), quantile(boot_wr, 0.95),
            mean(boot_wr > 0.524), mean(boot_roi > 0), perm_pvalue)
)
write_csv(results, here("results", "bootstrap_validation.csv"))
write_csv(thresh_curve, here("results", "threshold_stability.csv"))

cat("\nDone.\n")
