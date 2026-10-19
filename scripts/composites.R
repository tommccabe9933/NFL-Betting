# Composite / engineered features added on top of 19c's output.
#
# These were first built with one-off commands in April 2026 (they never lived
# in a script). The originals standardized with scale() over every season in
# the table, so adding a season silently changed every historical value.
# Here the mean/sd of each scaled input is frozen at the values from the
# original 2012-2025 table (objects/composite_scaling.rds), which reproduces
# modeling_data_v3_FINAL exactly and keeps new seasons from leaking backward.

library(tidyverse)

# Inputs that the composites standardize, in the order the originals did.
scaled_inputs <- list(
  step1 = c("weather_temperature", "fav_ngs_aggressiveness_roll5", "fav_ftn_roll5_blitz",
            "fav_injury_impact_score", "dog_qb_sack_rate_roll5", "fav_def_success_rate_roll5",
            "turnover_diff", "elo_diff", "fav_off_epa_play_roll3", "fav_def_epa_play_roll3", "week"),
  step2 = c("market_blindspot_score", "fav_improving", "dist_to_key", "epa_volatility_dog",
            "elo_spread_residual"),
  step2b = c("sharp_proxy"),
  step3 = c("dog_ftn_roll5_blitz")
)

scaling_stats <- function(df, cols) {
  map(set_names(cols), ~ list(mean = mean(df[[.x]], na.rm = TRUE),
                              sd   = sd(df[[.x]], na.rm = TRUE)))
}

z <- function(x, name, scaling) (x - scaling[[name]]$mean) / scaling[[name]]$sd

add_composites_step1 <- function(df, s) {
  df |> mutate(
    mbs_temp   = -z(weather_temperature, "weather_temperature", s),
    mbs_agg    =  z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s),
    mbs_blitz  =  z(fav_ftn_roll5_blitz, "fav_ftn_roll5_blitz", s),
    mbs_injury =  z(fav_injury_impact_score, "fav_injury_impact_score", s),
    mbs_sack   = -z(dog_qb_sack_rate_roll5, "dog_qb_sack_rate_roll5", s),
    market_blindspot_score = rowMeans(cbind(mbs_temp, mbs_agg, mbs_blitz, mbs_injury, mbs_sack),
                                      na.rm = TRUE),
    cover_conviction =
      -0.047 * z(weather_temperature, "weather_temperature", s) +
       0.047 * z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s) +
       0.042 * -z(fav_def_success_rate_roll5, "fav_def_success_rate_roll5", s) +
       0.037 * z(fav_ftn_roll5_blitz, "fav_ftn_roll5_blitz", s) +
       0.025 * z(turnover_diff, "turnover_diff", s) +
       0.021 * z(fav_injury_impact_score, "fav_injury_impact_score", s) +
       0.026 * z(elo_diff, "elo_diff", s),
    fresh_momentum = rowMeans(cbind(
      z(fav_off_epa_play_roll3, "fav_off_epa_play_roll3", s),
      -z(fav_def_epa_play_roll3, "fav_def_epa_play_roll3", s),
      z(turnover_diff, "turnover_diff", s)), na.rm = TRUE),
    pressure_matchup = rowMeans(cbind(
      z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s),
      -z(dog_qb_sack_rate_roll5, "dog_qb_sack_rate_roll5", s),
      z(fav_ftn_roll5_blitz, "fav_ftn_roll5_blitz", s)), na.rm = TRUE),
    situational_edge = rowMeans(cbind(
      mbs_temp,
      z(fav_injury_impact_score, "fav_injury_impact_score", s),
      -z(week, "week", s)), na.rm = TRUE)
  )
}

add_engineered_step2 <- function(df) {
  df |> mutate(
    spread_is_3   = as.numeric(spread_fav == 3),
    spread_is_7   = as.numeric(spread_fav == 7),
    spread_on_key = as.numeric(spread_fav %in% c(3, 7, 10, 14)),
    dist_to_3     = abs(spread_fav - 3),
    dist_to_7     = abs(spread_fav - 7),
    dist_to_key   = pmin(abs(spread_fav - 3), abs(spread_fav - 7), abs(spread_fav - 10)),
    epa_volatility_fav = abs(fav_off_epa_play_roll3 - fav_off_epa_play_roll5),
    epa_volatility_dog = abs(dog_off_epa_play_roll3 - dog_off_epa_play_roll5),
    fav_improving   = fav_off_epa_play_roll3 - fav_off_epa_play_roll5,
    dog_improving   = dog_off_epa_play_roll3 - dog_off_epa_play_roll5,
    improvement_gap = fav_improving - dog_improving,
    elo_spread_gap      = abs(elo_spread_residual),
    elo_says_bigger_fav = as.numeric(elo_spread_residual > 0),
    cold_game       = as.numeric(weather_temperature < 40),
    cold_and_windy  = as.numeric(weather_temperature < 45 & weather_wind_mph > 12),
    extreme_weather = as.numeric(weather_temperature < 32 | weather_wind_mph > 20),
    high_total = as.numeric(over_under_line > 48),
    low_total  = as.numeric(over_under_line < 41)
  )
}

add_composites_step2 <- function(df, s) {
  df |> mutate(
    sharp_proxy = rowMeans(cbind(
      z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s),
      z(fav_ftn_roll5_blitz, "fav_ftn_roll5_blitz", s),
      -z(dog_qb_sack_rate_roll5, "dog_qb_sack_rate_roll5", s),
      z(fav_injury_impact_score, "fav_injury_impact_score", s),
      z(turnover_diff, "turnover_diff", s)), na.rm = TRUE),
    mbs_v2 =
      0.30 * z(market_blindspot_score, "market_blindspot_score", s) +
      0.25 * z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s) +
      0.20 * -z(weather_temperature, "weather_temperature", s) +
      0.15 * z(fav_injury_impact_score, "fav_injury_impact_score", s) +
      0.10 * z(turnover_diff, "turnover_diff", s),
    mbs_v3 = rowMeans(cbind(
      z(market_blindspot_score, "market_blindspot_score", s),
      z(fav_improving, "fav_improving", s),
      -z(dist_to_key, "dist_to_key", s)), na.rm = TRUE),
    everything_score = rowMeans(cbind(
      -z(weather_temperature, "weather_temperature", s),
      z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s),
      z(fav_ftn_roll5_blitz, "fav_ftn_roll5_blitz", s),
      z(fav_injury_impact_score, "fav_injury_impact_score", s),
      -z(dog_qb_sack_rate_roll5, "dog_qb_sack_rate_roll5", s),
      z(turnover_diff, "turnover_diff", s),
      z(fav_improving, "fav_improving", s),
      -z(dist_to_key, "dist_to_key", s),
      z(elo_spread_residual, "elo_spread_residual", s)), na.rm = TRUE)
  )
}

add_composites_step2b <- function(df, s) {
  df |> mutate(
    sharp_v2 = rowMeans(cbind(
      z(sharp_proxy, "sharp_proxy", s),
      z(fav_improving, "fav_improving", s),
      -z(epa_volatility_dog, "epa_volatility_dog", s)), na.rm = TRUE)
  )
}

add_composites_step3 <- function(df, s) {
  df |> mutate(
    fav_expected_score    = (over_under_line + spread_fav) / 2,
    dog_expected_score    = (over_under_line - spread_fav) / 2,
    total_spread_script   = fav_expected_score / pmax(dog_expected_score, 1),
    high_total_big_spread = as.numeric(over_under_line > 48 & spread_fav > 7),
    fav_cold_streak = as.numeric(fav_improving < -0.05),
    fav_hot_streak  = as.numeric(fav_improving > 0.05),
    dog_cold_streak = as.numeric(dog_improving < -0.05),
    dog_hot_streak  = as.numeric(dog_improving > 0.05),
    fade_dog_heat   = dog_hot_streak,
    buy_fav_dip     = fav_cold_streak,
    qb_scheme_fit   = z(fav_ngs_aggressiveness_roll5, "fav_ngs_aggressiveness_roll5", s) *
                      z(dog_ftn_roll5_blitz, "dog_ftn_roll5_blitz", s),
    fav_pass_lean    = fav_off_pass_epa_roll5 - fav_off_rush_epa_roll5,
    dog_def_weakness = dog_def_pass_epa_roll5 - dog_def_rush_epa_roll5,
    defensive_style_clash = fav_pass_lean * dog_def_weakness
  )
}

# Full chain. With scaling = NULL, stats are computed from df itself at each
# step (the original behaviour) and returned so they can be frozen.
add_all_composites <- function(df, scaling = NULL) {
  learn <- is.null(scaling)
  if (learn) scaling <- list()
  if (learn) scaling <- c(scaling, scaling_stats(df, scaled_inputs$step1))
  df <- add_composites_step1(df, scaling)
  df <- add_engineered_step2(df)
  if (learn) scaling <- c(scaling, scaling_stats(df, scaled_inputs$step2))
  df <- add_composites_step2(df, scaling)
  if (learn) scaling <- c(scaling, scaling_stats(df, scaled_inputs$step2b))
  df <- add_composites_step2b(df, scaling)
  if (learn) scaling <- c(scaling, scaling_stats(df, scaled_inputs$step3))
  df <- add_composites_step3(df, scaling)
  attr(df, "composite_scaling") <- scaling
  df
}
