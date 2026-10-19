# Calibrated Kelly Sizing — v3.5 combined portfolio
# Step 1: Walk-forward calibration of model probabilities
# Step 2: Bankroll simulation on calibrated probabilities (2023-2025)
# Step 3: Forward-looking risk for a 2026-sized season
# Step 4: Fit calibrators on 2023-2025 for live 2026 use
#
# Why: the raw ensemble probabilities are overconfident. Bets the model
# called ~0.82 won ~60% of the time, so Kelly sized off raw p overbets.
# Run 44_bootstrap_combined.R first (it saves the per-game predictions).

library(tidyverse); library(here)

set.seed(42)

payout     <- 100 / 110            # profit per $1 risked at -110
breakeven  <- 1 / (1 + payout)     # 0.5238
bank_start <- 10000
bet_cap    <- 0.05                 # max 5% of bankroll on one game
week_cap   <- 0.25                 # max 25% of bankroll at risk in one week
flat_pct   <- 0.01                 # burn-in / baseline stake

cover_preds   <- read_rds(here("objects", "cover_preds_v35.rds"))
nocover_preds <- read_rds(here("objects", "nocover_preds_v35.rds"))
portfolio     <- read_rds(here("objects", "combined_portfolio_v35.rds"))

cat("=== CALIBRATED KELLY SIZING ===\n\n")

# ================================================================
# STEP 1: Walk-forward Platt calibration
# Season s is calibrated on ALL games (not just bets) from earlier test
# seasons, so no season's outcomes are used to size its own bets.
# 2023 has no earlier out-of-sample predictions → flat-bet burn-in year.
# ================================================================

clip <- function(p) pmin(pmax(p, 0.01), 0.99)

fit_platt <- function(p, y) {
  glm(y ~ x, family = binomial, data = tibble(x = qlogis(clip(p)), y = y))
}

apply_platt <- function(fit, p) {
  unname(predict(fit, tibble(x = qlogis(clip(p))), type = "response"))
}

calibrate_walk_forward <- function(preds, p_col, y_col) {
  map_dfr(sort(unique(preds$season)), function(s) {
    past <- preds |> filter(season < s)
    cur  <- preds |> filter(season == s)
    if (nrow(past) < 200) return(cur |> mutate(p_cal = NA_real_))
    fit <- fit_platt(past[[p_col]], past[[y_col]])
    cur |> mutate(p_cal = apply_platt(fit, .data[[p_col]]))
  })
}

cover_cal   <- calibrate_walk_forward(cover_preds,   "p_cover_ens", "truth_cover")
nocover_cal <- calibrate_walk_forward(nocover_preds, "p_nocover",   "truth_nocover")

bets <- portfolio |>
  left_join(cover_cal   |> select(game_id_norm, season, p_cal_c  = p_cal),
            by = c("game_id_norm", "season")) |>
  left_join(nocover_cal |> select(game_id_norm, season, p_cal_nc = p_cal),
            by = c("game_id_norm", "season")) |>
  mutate(p_cal = if_else(bet_type == "cover", p_cal_c, p_cal_nc)) |>
  select(-p_cal_c, -p_cal_nc) |>
  arrange(season, week)

stopifnot(nrow(bets) == nrow(portfolio))

# Reliability of the bets themselves, raw vs calibrated (2024-2025 only)
oos <- bets |> filter(!is.na(p_cal))
cat("Reliability on 2024-2025 bets (out-of-sample calibration):\n")
oos |>
  mutate(bucket = cut(p_used, c(0, 0.55, 0.58, 0.62, 0.70, 1))) |>
  group_by(bucket) |>
  summarize(n = n(), raw_p = mean(p_used), calibrated_p = mean(p_cal),
            actual = mean(win), .groups = "drop") |>
  mutate(across(c(raw_p, calibrated_p, actual), ~round(.x, 3))) |>
  print()

brier_raw <- mean((oos$p_used - oos$win)^2)
brier_cal <- mean((oos$p_cal  - oos$win)^2)
cat(sprintf("\nBrier score (lower = better): raw %.4f | calibrated %.4f\n",
  brier_raw, brier_cal))
cat(sprintf("Mean predicted win prob: raw %.3f | calibrated %.3f | actual %.3f\n",
  mean(oos$p_used), mean(oos$p_cal), mean(oos$win)))
no_edge <- oos$p_cal <= breakeven
cat(sprintf("Bets with no calibrated edge (p_cal <= %.3f): %d of %d — won %.1f%% vs %.1f%% for the rest\n\n",
  breakeven, sum(no_edge), nrow(oos), 100 * mean(oos$win[no_edge]), 100 * mean(oos$win[!no_edge])))

# ================================================================
# STEP 2: Bankroll simulation
# Every bet in a week is sized off the bankroll at the start of that week
# (games are simultaneous), then the week's results are settled together.
# Drawdown = largest fall from a running peak to a later low.
# ================================================================

kelly_full <- function(p) (payout * p - (1 - p)) / payout

# stake_fn(bets_this_week) → stake as a fraction of bankroll for each bet
simulate_bankroll <- function(bets, stake_fn, return_path = FALSE) {
  bank <- bank_start; peak <- bank; max_dd <- 0
  n_placed <- 0; staked_pct <- c(); path <- c()
  weeks <- split(bets, list(bets$season, bets$week), drop = TRUE)
  weeks <- weeks[order(map_dbl(weeks, ~.x$season[1] * 100 + .x$week[1]))]
  for (wk in weeks) {
    f <- pmax(stake_fn(wk), 0)
    f <- pmin(f, bet_cap)
    if (sum(f) > week_cap) f <- f * week_cap / sum(f)
    stake <- f * bank
    bank  <- bank + sum(ifelse(wk$win, stake * payout, -stake))
    peak   <- max(peak, bank)
    max_dd <- max(max_dd, (peak - bank) / peak)
    n_placed   <- n_placed + sum(f > 0)
    staked_pct <- c(staked_pct, f[f > 0])
    path <- c(path, bank)
  }
  if (return_path) {
    return(tibble(season = map_dbl(weeks, ~.x$season[1]),
                  week   = map_dbl(weeks, ~.x$week[1]), bankroll = path))
  }
  tibble(final = bank, return_pct = 100 * (bank / bank_start - 1),
         max_dd_pct = 100 * max_dd, bets_placed = n_placed,
         avg_stake_pct = 100 * mean(staked_pct))
}

# Calibrated strategies flat-bet 2023 (no calibration data yet)
calibrated_kelly <- function(frac) {
  function(wk) if_else(is.na(wk$p_cal), flat_pct, frac * kelly_full(wk$p_cal))
}

strategies <- list(
  "Flat $110 per bet (reference)"         = NULL,
  "Flat 1% of bankroll"                   = function(wk) rep(flat_pct, nrow(wk)),
  "Raw-probability 1/4 Kelly (old way)"   = function(wk) 0.25 * kelly_full(wk$p_used),
  "Calibrated 1/10 Kelly"                 = calibrated_kelly(0.10),
  "Calibrated 1/4 Kelly"                  = calibrated_kelly(0.25),
  "Calibrated 1/2 Kelly"                  = calibrated_kelly(0.50),
  "Calibrated full Kelly"                 = calibrated_kelly(1.00)
)

flat_profit <- sum(bets$profit)
sim_table <- imap_dfr(strategies, function(fn, name) {
  if (is.null(fn)) {
    return(tibble(strategy = name, final = bank_start + flat_profit,
                  return_pct = 100 * flat_profit / bank_start,
                  max_dd_pct = NA_real_, bets_placed = nrow(bets),
                  avg_stake_pct = 100 * 110 / bank_start))
  }
  simulate_bankroll(bets, fn) |> mutate(strategy = name, .before = 1)
})

cat("=== BANKROLL SIMULATION 2023-2025 ($10,000 start, 2023 = flat 1% burn-in) ===\n")
sim_table |>
  mutate(across(c(final, return_pct, max_dd_pct, avg_stake_pct), ~round(.x, 1))) |>
  print(width = Inf)

bankroll_paths <- strategies |>
  discard(is.null) |>
  imap_dfr(~simulate_bankroll(bets, .x, return_path = TRUE) |> mutate(strategy = .y, .before = 1))

# ================================================================
# STEP 3: Forward-looking risk for one 2026-sized season
# Stake sizes come from 2025's bets calibrated with 2023-2025 calibrators.
# Each scenario assumes a true win rate; outcomes are simulated 5,000 times.
# ================================================================

cal_cover_final   <- fit_platt(cover_preds$p_cover_ens, cover_preds$truth_cover)
cal_nocover_final <- fit_platt(nocover_preds$p_nocover, nocover_preds$truth_nocover)

season_template <- bets |>
  filter(season == 2025) |>
  mutate(p_cal_final = if_else(bet_type == "cover",
    apply_platt(cal_cover_final, p_used), apply_platt(cal_nocover_final, p_used)))

scenarios <- c("No edge (52.4%)" = breakeven, "Half the backtest edge (54.2%)" = 0.542,
               "Backtest edge (56.0%)" = 0.560)
fracs <- c("Flat 1%" = NA, "1/10 Kelly" = 0.10, "1/4 Kelly" = 0.25,
           "1/2 Kelly" = 0.50, "Full Kelly" = 1.00)

risk_table <- expand_grid(scenario = names(scenarios), sizing = names(fracs)) |>
  pmap_dfr(function(scenario, sizing) {
    frac <- fracs[[sizing]]
    stake_fn <- if (is.na(frac)) function(wk) rep(flat_pct, nrow(wk)) else
      function(wk) frac * kelly_full(wk$p_cal_final)
    sims <- map_dfr(1:5000, function(i) {
      wk_bets <- season_template |> mutate(win = runif(n()) < scenarios[[scenario]])
      simulate_bankroll(wk_bets, stake_fn)
    })
    tibble(scenario, sizing,
           median_return_pct = median(sims$return_pct),
           p_lose_money      = mean(sims$final < bank_start),
           p_drawdown_25     = mean(sims$max_dd_pct >= 25),
           worst_5pct_return = quantile(sims$return_pct, 0.05))
  })

cat("\n=== ONE-SEASON RISK (164 bets, 5,000 simulations each) ===\n")
risk_table |>
  mutate(across(where(is.numeric), ~round(.x, 3))) |>
  print(n = Inf, width = Inf)

# ================================================================
# STEP 4: Save calibrators for 2026 and results
# ================================================================

calibrators <- list(
  cover   = coef(cal_cover_final),
  nocover = coef(cal_nocover_final),
  fit_on  = "2023-2025 out-of-sample predictions, all games",
  usage   = "p_cal = plogis(a + b * qlogis(clip(p_raw, 0.01, 0.99)))"
)
cat("\nCalibrators fit on 2023-2025 (for 2026 sizing):\n")
cat(sprintf("  cover:    a = %.3f, b = %.3f\n", calibrators$cover[1],   calibrators$cover[2]))
cat(sprintf("  no-cover: a = %.3f, b = %.3f\n", calibrators$nocover[1], calibrators$nocover[2]))

plot_df <- oos |>
  mutate(bucket = cut(p_used, c(0, 0.55, 0.58, 0.62, 0.70, 1)), raw = p_used) |>
  pivot_longer(c(raw, p_cal), names_to = "version", values_to = "p") |>
  mutate(version = if_else(version == "raw", "Raw model", "Calibrated")) |>
  group_by(version, bucket) |>
  summarize(predicted = mean(p), actual = mean(win), n = n(), .groups = "drop")

calibration_plot <- ggplot(plot_df, aes(predicted, actual, colour = version)) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = breakeven, linetype = "dotted", colour = "firebrick") +
  geom_line() + geom_point(aes(size = n)) +
  coord_cartesian(xlim = c(0.5, 0.9), ylim = c(0.4, 0.9)) +
  labs(title = "Bet win probability: predicted vs actual (2024-2025 bets)",
       subtitle = "Dashed = perfect calibration, dotted = -110 breakeven",
       x = "Predicted win probability", y = "Actual win rate",
       colour = NULL, size = "Bets") +
  theme_minimal()

ggsave(here("figures", "calibration_v35.png"), calibration_plot,
       width = 7, height = 5, dpi = 150)

write_rds(calibrators, here("objects", "calibrators_v35.rds"))
write_rds(bets,        here("objects", "combined_portfolio_v35_calibrated.rds"))
write_csv(sim_table,   here("results", "kelly_calibrated_v35.csv"))
write_csv(risk_table,  here("results", "kelly_risk_one_season.csv"))
write_csv(bankroll_paths, here("results", "kelly_bankroll_paths_v35.csv"))

cat("\nSaved: objects/calibrators_v35.rds\n")
cat("       objects/combined_portfolio_v35_calibrated.rds\n")
cat("       results/kelly_calibrated_v35.csv\n")
cat("       results/kelly_bankroll_paths_v35.csv\n")
cat("       results/kelly_risk_one_season.csv\n")
cat("       figures/calibration_v35.png\n")
