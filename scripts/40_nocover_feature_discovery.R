# No-Cover Feature Discovery
# Test ALL stable features against the no_cover target
# Exclude 37 flippy features — unreliable for both sides
# Find features where HIGH value = DOG COVERS

library(tidyverse)
library(here)

v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

# no_cover target: 1 when dog covers
nc_target <- as.integer(v3$fav_cover == "no_cover")
# cover target for comparison
c_target <- as.integer(v3$fav_cover == "cover")

cat("=== NO-COVER FEATURE DISCOVERY ===\n")
cat(sprintf("Games: %d | No-cover rate: %.1f%%\n\n", nrow(v3), mean(nc_target)*100))

# Exclude flippers
flip_features <- c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff","rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa","injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged","fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3","fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3","off_success_rate_roll5_diff","def_pass_epa_roll3_diff","fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff","fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5","fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5","dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff","off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff","def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
bad_new <- c("total_spread_script","high_total_big_spread","dog_cold_streak","dog_hot_streak","fade_dog_heat","qb_scheme_fit","defensive_style_clash","fav_expected_score","dog_expected_score","fav_pass_lean","dog_def_weakness")

stable_feats <- setdiff(v3 |> select(where(is.numeric), -season, -week) |> names(), c(flip_features, bad_new))
cat("Stable features:", length(stable_feats), "\n\n")

# Correlate all features with NO-COVER target
cat("=== ALL FEATURES vs NO-COVER (positive = dog covers more) ===\n\n")

nc_cors <- map_dfr(stable_feats, function(f) {
  x <- v3[[f]]; valid <- !is.na(x)
  if (sum(valid) < 500) return(NULL)
  nc_cor <- cor(x[valid], nc_target[valid])
  c_cor <- cor(x[valid], c_target[valid])
  spread_cor <- cor(x[valid], v3$spread_fav[valid])

  # Check per-season stability
  s23 <- v3$season == 2023 & valid; s24 <- v3$season == 2024 & valid; s25 <- v3$season == 2025 & valid
  cor_23 <- if (sum(s23)>50) cor(x[s23], nc_target[s23]) else NA
  cor_24 <- if (sum(s24)>50) cor(x[s24], nc_target[s24]) else NA
  cor_25 <- if (sum(s25)>50) cor(x[s25], nc_target[s25]) else NA

  # Stable = same sign in 2+ of 3 test seasons
  signs <- c(sign(cor_23), sign(cor_24), sign(cor_25))
  signs <- signs[!is.na(signs)]
  stable <- length(signs) >= 2 & (sum(signs > 0) >= 2 | sum(signs < 0) >= 2)

  tibble(feature=f, nc_cor=nc_cor, c_cor=c_cor, spread_cor=spread_cor,
         cor_23=cor_23, cor_24=cor_24, cor_25=cor_25, stable=stable, n=sum(valid))
}) |> arrange(desc(abs(nc_cor)))

# Top 30 no-cover features
cat("Top 30 features for NO-COVER:\n")
cat(sprintf("%-40s %7s %7s %7s %7s %7s %7s %6s\n",
  "Feature", "NC_cor", "C_cor", "SprCor", "2023", "2024", "2025", "Stable"))
cat(paste(rep("-", 95), collapse=""), "\n")

for (i in 1:min(30, nrow(nc_cors))) {
  r <- nc_cors[i,]
  cat(sprintf("%-40s %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %5s\n",
    r$feature, r$nc_cor, r$c_cor, r$spread_cor,
    ifelse(is.na(r$cor_23), 0, r$cor_23),
    ifelse(is.na(r$cor_24), 0, r$cor_24),
    ifelse(is.na(r$cor_25), 0, r$cor_25),
    if (r$stable) "YES" else "no"))
}

# Features that are STABLE for no-cover AND market-orthogonal
cat("\n=== STABLE NO-COVER FEATURES (same sign 2+ seasons, |spread_cor| < 0.15) ===\n")
stable_nc <- nc_cors |> filter(stable, abs(nc_cor) > 0.03, abs(spread_cor) < 0.15)
cat(sprintf("Found %d stable market-orthogonal no-cover features:\n\n", nrow(stable_nc)))
for (i in 1:min(20, nrow(stable_nc))) {
  r <- stable_nc[i,]
  cat(sprintf("  %-40s nc_cor=%+.4f spread=%+.4f [%+.3f, %+.3f, %+.3f]\n",
    r$feature, r$nc_cor, r$spread_cor,
    ifelse(is.na(r$cor_23),0,r$cor_23), ifelse(is.na(r$cor_24),0,r$cor_24), ifelse(is.na(r$cor_25),0,r$cor_25)))
}

# KEY: features that predict NO-COVER but NOT COVER (asymmetric)
cat("\n=== ASYMMETRIC FEATURES (predict nc but NOT cover) ===\n")
asym <- nc_cors |> filter(abs(nc_cor) > 0.03, abs(c_cor) < 0.02)
cat(sprintf("Found %d asymmetric features:\n", nrow(asym)))
if (nrow(asym) > 0) {
  for (i in 1:min(10, nrow(asym))) {
    r <- asym[i,]
    cat(sprintf("  %-40s nc=%+.4f c=%+.4f (asymmetric!)\n", r$feature, r$nc_cor, r$c_cor))
  }
}

write_csv(nc_cors, here("results", "nocover_feature_correlations.csv"))
cat("\nDone.\n")
