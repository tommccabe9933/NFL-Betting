# New composite features — written down with their reasoning before testing
#
# Each composite combines related signals into one number. Inputs are
# z-scored with 2012-2022 stats only. Positive values are meant to favor
# the FAVORITE covering. Tested as plain correlations with cover:
#   pooled 2016-2025 (~2,600 games), each season, partial r after removing
#   the spread/total/Elo, and fresh 2026 games (never examined before).
# With 10 composites and ~2,600 games, |r| needs to be above ~0.055 to beat
# luck after accounting for testing 10 things.

library(tidyverse); library(here)

v <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(!is.na(fav_cover), season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover"))

ref <- v |> filter(season <= 2022)
z <- function(x, name) (x - mean(ref[[name]], na.rm = TRUE)) / sd(ref[[name]], na.rm = TRUE)
zc <- function(col) z(v[[col]], col)
rm2 <- function(...) rowMeans(cbind(...), na.rm = TRUE)

v <- v |> mutate(
  # 1. Total matchup edge: each offense against the other defense, pass + run
  net_matchup = rm2(
    zc("fav_off_pass_epa_roll5") - zc("dog_def_pass_epa_roll5"),
    zc("fav_off_rush_epa_roll5") - zc("dog_def_rush_epa_roll5"),
    -(zc("dog_off_pass_epa_roll5") - zc("fav_def_pass_epa_roll5")),
    -(zc("dog_off_rush_epa_roll5") - zc("fav_def_rush_epa_roll5"))),
  # 2. Injury burden: dog more banged up than fav (impact + players out)
  injury_net = rm2(zc("injury_diff"), -zc("fav_n_out"), zc("dog_n_out")),
  # 3. QB edge: efficiency + accuracy - getting sacked
  qb_edge = rm2(zc("qb_epa_per_play_roll5_diff"), zc("qb_cpoe_roll5_diff"), -zc("qb_sack_rate_roll5_diff")),
  # 4. Trench edge: fav pass rush vs dog protection, minus the reverse
  trench_edge = rm2(zc("fav_pfr_roll5_def_pressure"), zc("dog_pfr_roll5_pressure_rate"),
                    -zc("dog_pfr_roll5_def_pressure"), -zc("fav_pfr_roll5_pressure_rate")),
  # 5. Stats vs price: EPA edge bigger than the spread implies
  stats_vs_spread = zc("off_epa_play_roll5_diff") - zc("def_epa_play_roll5_diff") - zc("spread_fav"),
  # 6. Elo vs price plus recent form: a favorite the market may be underrating
  underpriced_fav = rm2(zc("elo_spread_residual"), zc("improvement_gap")),
  # 7. Turnover luck: big turnover margins tend to regress, so expect negative
  turnover_luck = -zc("turnover_diff"),
  # 8. Chaos x spread: wind/cold make big spreads harder to cover, expect negative
  chaos_x_spread = (zc("weather_wind_mph") - zc("weather_temperature")) * zc("spread_fav"),
  # 9. Rest edge, amplified when the favorite is home
  rest_home = zc("rest_diff") * if_else(fav_at_home == "yes", 1.5, 1),
  # 10. Consistency: steadier favorite vs more volatile dog
  steadiness = zc("epa_volatility_dog") - zc("epa_volatility_fav")
)
composites <- c("net_matchup", "injury_net", "qb_edge", "trench_edge", "stats_vs_spread",
                "underpriced_fav", "turnover_luck", "chaos_x_spread", "rest_home", "steadiness")

r <- function(d, f) {
  ok <- is.finite(d[[f]]); if (sum(ok) < 40) return(NA_real_)
  cor(d[[f]][ok], d$y[ok])
}
partial_r <- function(d, f) {
  d <- d |> filter(is.finite(.data[[f]]), !is.na(elo_diff))
  ctrl <- lm(cbind(d[[f]], d$y) ~ spread_fav + over_under_line + elo_diff, data = d)
  cor(resid(ctrl)[, 1], resid(ctrl)[, 2])
}

hist <- v |> filter(season >= 2016, season <= 2025)
fresh <- v |> filter(season == 2026)

res <- map_dfr(composites, function(f) {
  by_season <- map_dbl(2016:2025, ~ r(filter(hist, season == .x), f))
  tibble(composite = f,
         r_2016_25 = r(hist, f),
         partial_r = partial_r(hist, f),
         seasons_positive = sum(by_season > 0, na.rm = TRUE),
         r_2023_25 = r(filter(hist, season >= 2023), f),
         r_2026_fresh = r(fresh, f),
         n = sum(is.finite(hist[[f]])))
}) |> arrange(desc(abs(r_2016_25)))

cat(sprintf("Games: 2016-2025 = %d | 2026 fresh = %d\n", nrow(hist), nrow(fresh)))
cat("Luck bar: |r| > ~0.055 pooled (10 tests); 2026 alone is too small to judge (±0.21)\n\n")
print(res |> mutate(across(where(is.double), ~ round(.x, 3))), width = Inf)
write_csv(res, here("results", "new_composites_v37.csv"))
