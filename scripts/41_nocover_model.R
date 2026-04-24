# No-Cover Model — train directly on no_cover target
# Use stable features, exclude flippers, separate model from cover

library(tidyverse); library(here); library(glmnet)
source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

flip_features <- c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff","rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa","injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged","fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3","fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3","off_success_rate_roll5_diff","def_pass_epa_roll3_diff","fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff","fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5","fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5","dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff","off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff","def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
bad_new <- c("total_spread_script","high_total_big_spread","dog_cold_streak","dog_hot_streak","fade_dog_heat","qb_scheme_fit","defensive_style_clash","fav_expected_score","dog_expected_score","fav_pass_lean","dog_def_weakness")
stable_feats <- setdiff(v3 |> select(where(is.numeric), -season, -week) |> names(), c(flip_features, bad_new))

# First add no-cover specific features
v3 <- v3 |> mutate(
  # Market overextension (spread > Elo implies)
  spread_overextension = spread_fav - elo_implied_spread,
  # Variance environment
  variance_env = rowMeans(cbind(
    scale(weather_wind_mph)[,1],
    -scale(weather_temperature)[,1],
    scale(over_under_line)[,1]
  ), na.rm = TRUE),
  # Dog momentum (dog improving, fav fading)
  dog_momentum_edge = dog_improving - fav_improving,
  # Favorite overpriced (results > process)
  fav_overpriced = fav_roll5_pts - fav_off_epa_play_roll5 * 15,  # scale EPA to points
  # Dog underestimated (process > results)
  dog_underestimated = dog_off_epa_play_roll5 * 15 - dog_roll5_pts,
  # No-cover composite (mirror of cover_conviction but for dog)
  nocover_conviction = rowMeans(cbind(
    scale(spread_overextension)[,1],
    scale(weather_temperature)[,1],  # warm = more no-cover
    -scale(fav_ngs_aggressiveness_roll5)[,1],  # passive QB = no-cover
    scale(fav_ftn_roll5_rpo)[,1],  # RPO-heavy = no-cover
    scale(fav_cold_streak)[,1]  # cold fav = no-cover
  ), na.rm = TRUE)
)

# Add new features to stable set
stable_feats <- c(stable_feats, "spread_overextension", "variance_env", "dog_momentum_edge",
                   "fav_overpriced", "dog_underestimated", "nocover_conviction")

cat("=== NO-COVER MODEL ===\n")
cat("Features:", length(stable_feats), "\n\n")

# Quick check new feature correlations
nc_target <- as.integer(v3$fav_cover == "no_cover")
for (f in c("spread_overextension","variance_env","dog_momentum_edge",
            "fav_overpriced","dog_underestimated","nocover_conviction")) {
  x <- v3[[f]]; valid <- !is.na(x)
  if (sum(valid) < 500) next
  cat(sprintf("  %-25s nc_cor=%+.4f spread_cor=%+.4f n=%d\n",
    f, cor(x[valid], nc_target[valid]), cor(x[valid], v3$spread_fav[valid]), sum(valid)))
}

# Train no-cover model with FLIPPED target
# Target: "no_cover" = first level (dogs cover)
run_nc_model <- function(start, alpha, lambda) {
  all_p <- list()
  for (ts in c(2023, 2024, 2025)) {
    tr <- v3 |> filter(season >= start, season < ts)
    te <- v3 |> filter(season == ts)
    if (nrow(tr) < 50) next

    tc <- map_dbl(stable_feats, ~mean(!is.na(tr[[.x]]))); fe <- stable_feats[tc > 0.5]
    tx <- tr|>select(all_of(fe)); md <- map_dbl(tx, ~median(.x, na.rm=T))
    tx <- tx|>mutate(across(everything(), ~replace_na(.x, md[cur_column()])))|>as.matrix()
    ex <- te|>select(all_of(fe))|>mutate(across(everything(), ~replace_na(.x, md[cur_column()])))|>as.matrix()
    m <- colMeans(tx); s <- apply(tx,2,sd); s[s==0] <- 1
    txs <- scale(tx,m,s); exs <- scale(ex,m,s)
    gc <- which(apply(txs,2,sd)>0); txs <- txs[,gc]; exs <- exs[,gc]

    # Target: no_cover first, cover second
    y <- factor(tr$fav_cover, levels = c("no_cover", "cover"))
    fit <- glmnet(txs, y, family="binomial", alpha=alpha, lambda=lambda)
    # predict returns P(second level) = P(cover)
    # So P(no_cover) = 1 - predict
    pred <- 1 - predict(fit, exs, type="response")[,1]

    all_p[[as.character(ts)]] <- tibble(
      game_id_norm=te$game_id_norm, p_nocover=pred, test_season=ts,
      fav_cover=te$fav_cover, week=te$week, spread=te$spread_fav
    )
  }
  bind_rows(all_p)
}

cat("\n=== TRAINING NO-COVER MODELS ===\n")

# Test multiple configs
results <- list()
for (start in c(2016, 2020, 2022)) {
  for (alpha in c(0, 0.1, 0.5)) {
    for (lambda in c(0.005, 0.01, 0.05, 0.10)) {
      preds <- run_nc_model(start, alpha, lambda)
      if (nrow(preds) == 0) next

      truth_nc <- preds$fav_cover == "no_cover"

      for (th in c(0.52, 0.53, 0.54, 0.55, 0.56, 0.57)) {
        bets <- preds |> filter(p_nocover > th)
        if (nrow(bets) < 15) next
        wr <- mean(bets$fav_cover == "no_cover")
        profit <- sum(ifelse(bets$fav_cover == "no_cover", 100, -110))

        results[[length(results)+1]] <- tibble(
          start=start, alpha=alpha, lambda=lambda, threshold=th,
          n_bets=nrow(bets), nc_wr=wr, profit=profit
        )
      }
    }
  }
}

res <- bind_rows(results)
cat(sprintf("Total configs tested: %d\n\n", nrow(res)))

# Best by profit (>= 30 bets)
cat("=== TOP 10 NO-COVER CONFIGS BY PROFIT (>= 30 bets) ===\n")
top <- res |> filter(n_bets >= 30) |> arrange(desc(profit)) |> head(10)
for (i in 1:nrow(top)) {
  r <- top[i,]
  cat(sprintf("  train=%d a=%.1f lam=%.3f th=%.2f: %d bets %.1f%% $%d ROI:%.1f%%\n",
    r$start, r$alpha, r$lambda, r$threshold, r$n_bets, r$nc_wr*100, r$profit,
    r$profit/(r$n_bets*110)*100))
}

# Check per-season for best config
if (nrow(top) > 0) {
  best <- top[1,]
  cat(sprintf("\n=== BEST CONFIG BY SEASON: train=%d a=%.1f lam=%.3f th=%.2f ===\n",
    best$start, best$alpha, best$lambda, best$threshold))

  preds <- run_nc_model(best$start, best$alpha, best$lambda)
  bets <- preds |> filter(p_nocover > best$threshold)
  bets |> group_by(test_season) |>
    summarize(n=n(), nc_wr=round(mean(fav_cover=="no_cover")*100,1),
              profit=sum(ifelse(fav_cover=="no_cover",100,-110)), .groups="drop") |>
    print()

  # Check: does no-cover activate in DIFFERENT games than cover?
  cat("\n=== INDEPENDENCE FROM COVER MODEL ===\n")
  cover_preds <- read_rds(here("objects", "ensemble_3model_v2.rds"))

  if (nrow(cover_preds) > 0) {
    # Which games does cover bet on? (using our balanced algo approximation)
    cover_bets <- cover_preds |>
      mutate(cover_bet = (pA > 0.55) & (pB > 0.55) & (pC > 0.55)) |>
      filter(cover_bet) |>
      pull(game_id_norm)

    nc_bets <- bets$game_id_norm
    overlap <- sum(nc_bets %in% cover_bets)
    cat(sprintf("  Cover bets: %d | No-cover bets: %d | Overlap: %d\n",
      length(cover_bets), length(nc_bets), overlap))
    cat(sprintf("  Overlap rate: %.1f%% — %s\n",
      overlap/max(length(nc_bets),1)*100,
      if (overlap/max(length(nc_bets),1) < 0.10) "GOOD (independent)" else "WARNING (correlated)"))
  }
}

write_rds(v3, here("objects", "modeling_data_v4.rds"))
write_csv(res, here("results", "nocover_model_results.csv"))
cat("\nDone.\n")
