# Bootstrap Validation — Combined Cover + No-Cover Portfolio
# Pre-registration: 2023-2024 strategy selection, 2025 = blind holdout
# Bootstrap B=2000, permutation test, sensitivity analysis

library(tidyverse); library(here); library(glmnet)
source(here("scripts", "00_project_setup.R"))

set.seed(42)
# MODEL_DATA / TAG let the same backtest run on the v3.6 rebuild
data_file <- Sys.getenv("MODEL_DATA", "modeling_data_v3_FINAL.rds")
tag <- Sys.getenv("TAG", "v35")
v3 <- read_rds(here("objects", data_file)) |> filter(!is.na(fav_cover))

cat("=== COMBINED PORTFOLIO BOOTSTRAP VALIDATION ===\n")
cat(sprintf("Dataset: %d x %d | Seasons: %d-%d\n\n",
  nrow(v3), ncol(v3), min(v3$season), max(v3$season)))

# ================================================================
# STEP 1: Regenerate cover and no-cover predictions (walk-forward)
# ================================================================

source(here("scripts", "model_functions.R"))
sf <- stable_features(v3)

make_preds <- function(data, start_yr, alpha_val, lambda_val, target_cover = TRUE) {
  map_dfr(c(2023, 2024, 2025), function(ts) {
    tr <- data |> filter(season >= start_yr, season < ts)
    te <- data |> filter(season == ts)
    tibble(game_id_norm = te$game_id_norm,
           p = fit_predict(tr, te, sf, alpha_val, lambda_val, target_cover),
           fav_cover = te$fav_cover, season = ts, week = te$week)
  })
}

cat("Generating cover predictions (Balanced ensemble)...\n")
# Balanced ensemble: A=train2022 a=0 lam=0.10, B=train2022 a=0 lam=0.005, C=train2022 a=0.5 lam=0.005
pA <- make_preds(v3, 2022, 0,   0.10, TRUE)
pB <- make_preds(v3, 2022, 0,   0.005, TRUE)
pC <- make_preds(v3, 2022, 0.5, 0.005, TRUE)

cover_preds <- pA |>
  rename(pA = p) |>
  inner_join(pB |> rename(pB = p) |> select(game_id_norm, season, pB), by = c("game_id_norm","season")) |>
  inner_join(pC |> rename(pC = p) |> select(game_id_norm, season, pC), by = c("game_id_norm","season")) |>
  mutate(p_cover_ens = (pA + pB + pC) / 3,
         truth_cover = fav_cover == "cover")

cat(sprintf("Cover preds: %d games, mean P=%.3f\n", nrow(cover_preds), mean(cover_preds$p_cover_ens)))

cat("Generating no-cover predictions...\n")
pNC <- make_preds(v3, 2012, 0.25, 0.08, FALSE)
nocover_preds <- pNC |>
  rename(p_nocover = p) |>
  mutate(truth_nocover = fav_cover == "no_cover")

cat(sprintf("No-cover preds: %d games, mean P=%.3f\n", nrow(nocover_preds), mean(nocover_preds$p_nocover)))

# ================================================================
# STEP 2: Build combined portfolio (resolve overlaps)
# ================================================================

cover_bets   <- cover_preds   |> filter(p_cover_ens > 0.55)
nocover_bets <- nocover_preds |> filter(p_nocover   > 0.53)

# Overlaps: same game bet by both models → take higher confidence
overlap_games <- intersect(cover_bets$game_id_norm, nocover_bets$game_id_norm)
cat(sprintf("\nOverlap: %d games → resolve by confidence\n", length(overlap_games)))

cover_keep <- cover_bets |>
  left_join(nocover_bets |> select(game_id_norm, p_nocover), by = "game_id_norm") |>
  filter(is.na(p_nocover) | p_cover_ens >= p_nocover) |>
  select(-p_nocover) |>
  mutate(bet_type = "cover", p_used = p_cover_ens,
         win = truth_cover, profit = ifelse(win, 100, -110))

nocover_keep <- nocover_bets |>
  left_join(cover_bets |> select(game_id_norm, p_cover_ens), by = "game_id_norm") |>
  filter(is.na(p_cover_ens) | p_nocover > p_cover_ens) |>
  select(-p_cover_ens) |>
  mutate(bet_type = "no_cover", p_used = p_nocover,
         win = truth_nocover, profit = ifelse(win, 100, -110))

portfolio <- bind_rows(
  cover_keep   |> select(game_id_norm, season, week, bet_type, p_used, win, profit),
  nocover_keep |> select(game_id_norm, season, week, bet_type, p_used, win, profit)
) |> arrange(season, week)

cat(sprintf("\n=== PORTFOLIO SUMMARY ===\n"))
cat(sprintf("Total bets: %d | Cover: %d | No-cover: %d\n",
  nrow(portfolio), sum(portfolio$bet_type == "cover"),
  sum(portfolio$bet_type == "no_cover")))
cat(sprintf("Win rate: %.1f%% | Profit: $%d | ROI: %.1f%%\n",
  mean(portfolio$win) * 100,
  sum(portfolio$profit),
  sum(portfolio$profit) / (nrow(portfolio) * 110) * 100))

cat("\nBy season:\n")
portfolio |>
  group_by(season) |>
  summarize(n = n(),
            cover_n = sum(bet_type == "cover"),
            nc_n    = sum(bet_type == "no_cover"),
            wr      = round(mean(win) * 100, 1),
            profit  = sum(profit), .groups = "drop") |>
  print()

cat("\nBy bet type:\n")
portfolio |>
  group_by(bet_type) |>
  summarize(n = n(), wr = round(mean(win)*100,1), profit = sum(profit), .groups="drop") |>
  print()

# ================================================================
# STEP 3: Cover-only baseline for comparison
# ================================================================
cover_only <- cover_keep |>
  summarize(n = n(), wr = mean(win), profit = sum(profit))
cat(sprintf("\nCover-only: %d bets, %.1f%%, $%d\n",
  cover_only$n, cover_only$wr * 100, cover_only$profit))
cat(sprintf("Combined:   %d bets, %.1f%%, $%d\n",
  nrow(portfolio), mean(portfolio$win)*100, sum(portfolio$profit)))

# ================================================================
# STEP 4: Kelly bankroll simulation — moved to 45_calibrated_kelly.R
# ================================================================
# The old simulation here sized bets off the raw model probabilities, which
# are overconfident above ~0.62, and computed drawdown as (peak - lowest
# point ever) / peak even when the low came before the peak. 45 calibrates
# the probabilities walk-forward first and tracks drawdown properly.
write_rds(cover_preds,   here("objects", str_glue("cover_preds_{tag}.rds")))
write_rds(nocover_preds, here("objects", str_glue("nocover_preds_{tag}.rds")))

# ================================================================
# STEP 5: Bootstrap CI (B=2000, stratified by season)
# ================================================================
cat("\n=== BOOTSTRAP VALIDATION (B=2000 stratified by season) ===\n")

B <- 2000
obs_wr     <- mean(portfolio$win)
obs_profit <- sum(portfolio$profit)

boot_wr <- numeric(B)
boot_pf <- numeric(B)

for (b in seq_len(B)) {
  boot_bets <- portfolio |>
    group_by(season) |>
    slice_sample(prop = 1, replace = TRUE) |>
    ungroup()
  boot_wr[b] <- mean(boot_bets$win)
  boot_pf[b] <- sum(boot_bets$profit)
}

ci_wr  <- quantile(boot_wr, c(0.05, 0.95))
ci_pf  <- quantile(boot_pf, c(0.05, 0.95))
p_beat <- mean(boot_wr > 0.524)  # P(above breakeven)

cat(sprintf("  Observed win rate: %.3f (%.1f%%)\n", obs_wr, obs_wr*100))
cat(sprintf("  90%% CI win rate:   [%.3f, %.3f]\n", ci_wr[1], ci_wr[2]))
cat(sprintf("  P(above breakeven): %.1f%%\n", p_beat*100))
cat(sprintf("  Observed profit:  $%d\n", round(obs_profit)))
cat(sprintf("  90%% CI profit:    [$%d, $%d]\n", round(ci_pf[1]), round(ci_pf[2])))

# ================================================================
# STEP 6: Permutation test — shuffle outcomes GLOBALLY (not within season)
# This tests: is our model selecting games above chance?
# Null: each bet wins at exactly breakeven (52.4%) — shuffle all outcomes
# ================================================================
cat("\n=== PERMUTATION TEST (global shuffle, N=2000) ===\n")
cat("  H0: bets win at exactly 52.4% (breakeven) | H1: win rate > 52.4%\n")

n_bets <- nrow(portfolio)
perm_profits <- numeric(2000)
all_outcomes <- portfolio$win  # observed win/loss outcomes

for (p_iter in seq_len(2000)) {
  # Simulate null: each bet wins at p=0.524 (breakeven probability)
  null_wins <- rbinom(n_bets, 1, prob = 0.524)
  perm_profits[p_iter] <- sum(ifelse(null_wins == 1, 100, -110))
}
perm_p <- mean(perm_profits >= obs_profit)
cat(sprintf("  Null simulation p-value: %.3f (%s)\n", perm_p,
  if(perm_p < 0.05) "SIGNIFICANT" else if(perm_p < 0.15) "marginal" else "not significant"))
cat(sprintf("  Null median profit: $%d | Observed: $%d\n",
  round(median(perm_profits)), round(obs_profit)))

# Also: binomial test — how unlikely is our win rate under H0: p=0.524?
binom_p <- binom.test(sum(portfolio$win), n_bets, p = 0.524, alternative = "greater")$p.value
cat(sprintf("  Binomial p (one-sided vs 52.4%%): %.4f (%s)\n", binom_p,
  if(binom_p < 0.05) "SIGNIFICANT" else if(binom_p < 0.10) "marginal" else "not significant"))

# ================================================================
# STEP 7: PRE-REGISTRATION — 2023-2024 selection, 2025 blind holdout
# ================================================================
cat("\n=== PRE-REGISTRATION FRAMEWORK ===\n")
cat("Strategy selected on 2023-2024 ONLY, 2025 = blind test\n\n")

dev_port  <- portfolio |> filter(season < 2025)
hold_port <- portfolio |> filter(season == 2025)

cat(sprintf("Development (2023-2024): %d bets, %.1f%%, $%d\n",
  nrow(dev_port), mean(dev_port$win)*100, sum(dev_port$profit)))
cat(sprintf("Holdout     (2025):       %d bets, %.1f%%, $%d\n",
  nrow(hold_port), mean(hold_port$win)*100, sum(hold_port$profit)))
cat(sprintf("2025 vs breakeven: %s (52.4%% needed)\n",
  if(mean(hold_port$win) > 0.524) "ABOVE" else "BELOW"))

# Bootstrap 2025 separately
boot_2025 <- numeric(2000)
for (b in seq_len(2000)) {
  s <- sample(hold_port$win, replace = TRUE)
  boot_2025[b] <- mean(s)
}
ci_2025 <- quantile(boot_2025, c(0.05, 0.95))
cat(sprintf("2025 90%% CI: [%.1f%%, %.1f%%]\n", ci_2025[1]*100, ci_2025[2]*100))

# ================================================================
# STEP 8: Sensitivity analysis
# ================================================================
cat("\n=== SENSITIVITY ANALYSIS ===\n")
cat("Varying cover threshold ±0.02 and no-cover threshold ±0.02\n\n")

cat(sprintf("%-12s %-12s %6s %6s %8s\n", "CoverTh", "NCTh", "Bets", "WR%", "Profit"))
cat(paste(rep("-", 50), collapse=""), "\n")

for (c_th in c(0.53, 0.55, 0.57)) {
  for (nc_th in c(0.51, 0.53, 0.55)) {
    c_bets  <- cover_preds   |> filter(p_cover_ens > c_th)
    nc_bets <- nocover_preds |> filter(p_nocover   > nc_th)
    ol_g    <- intersect(c_bets$game_id_norm, nc_bets$game_id_norm)

    c_k  <- c_bets  |> filter(!game_id_norm %in% ol_g |
                                (game_id_norm %in% ol_g & p_cover_ens >=
                                   (nc_bets$p_nocover[match(game_id_norm, nc_bets$game_id_norm)]))) |>
      mutate(win = fav_cover == "cover", profit = ifelse(win, 100, -110))
    nc_k <- nc_bets |> filter(!game_id_norm %in% ol_g |
                                (game_id_norm %in% ol_g & p_nocover >
                                   (c_bets$p_cover_ens[match(game_id_norm, c_bets$game_id_norm)]))) |>
      mutate(win = fav_cover == "no_cover", profit = ifelse(win, 100, -110))

    comb <- bind_rows(
      c_k  |> select(win, profit),
      nc_k |> select(win, profit)
    )
    if (nrow(comb) == 0) next
    cat(sprintf("  c>%.2f  nc>%.2f  %5d %5.1f%% $%6d\n",
      c_th, nc_th, nrow(comb), mean(comb$win)*100, sum(comb$profit)))
  }
}

# ================================================================
# STEP 9: Month-of-season stability (early vs late)
# ================================================================
cat("\n=== STABILITY BY GAME PHASE ===\n")
portfolio |>
  mutate(phase = case_when(
    week <= 4  ~ "Early (wk 1-4)",
    week <= 10 ~ "Mid   (wk 5-10)",
    week <= 18 ~ "Late  (wk 11-18)",
    TRUE       ~ "Playoffs"
  )) |>
  group_by(phase) |>
  summarize(n = n(), wr = round(mean(win)*100,1), profit = sum(profit), .groups = "drop") |>
  arrange(phase) |> print()

# ================================================================
# Save results
# ================================================================
results_summary <- tibble(
  metric           = c("n_bets","cover_bets","nocover_bets",
                       "win_rate","profit","roi_pct",
                       "boot_ci_lo","boot_ci_hi","p_above_breakeven",
                       "perm_pvalue",
                       "dev_wr","dev_profit","holdout_wr","holdout_profit",
                       "holdout_ci_lo","holdout_ci_hi"),
  value            = c(nrow(portfolio), sum(portfolio$bet_type=="cover"),
                       sum(portfolio$bet_type=="no_cover"),
                       obs_wr, obs_profit,
                       obs_profit/(nrow(portfolio)*110)*100,
                       ci_wr[1], ci_wr[2], p_beat,
                       perm_p,
                       mean(dev_port$win), sum(dev_port$profit),
                       mean(hold_port$win), sum(hold_port$profit),
                       ci_2025[1], ci_2025[2])
)

write_rds(portfolio,        here("objects", str_glue("combined_portfolio_{tag}.rds")))
write_csv(results_summary,  here("results", str_glue("bootstrap_combined_{tag}.csv")))

cat("\n=== BOOTSTRAP COMPLETE ===\n")
cat(sprintf("Portfolio: %d bets | %.1f%% WR | $%d profit\n",
  nrow(portfolio), obs_wr*100, round(obs_profit)))
cat(sprintf("90%% CI: [%.1f%%, %.1f%%] | P(>BE): %.0f%%\n",
  ci_wr[1]*100, ci_wr[2]*100, p_beat*100))
cat(sprintf("Permutation p = %.3f\n", perm_p))
cat(sprintf("2025 holdout: %.1f%% (%s)\n",
  mean(hold_port$win)*100, if(mean(hold_port$win)>0.524)"PROFITABLE" else "below BE"))
cat("\nSaved: objects/combined_portfolio_v35.rds\n")
cat("       results/bootstrap_combined_v35.csv\n")
