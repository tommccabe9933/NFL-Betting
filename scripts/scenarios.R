# Scenario conditions shared by 54_scenario_search.R and 55_scenario_tracker.R
# Cutoffs (terciles, injury z-score stats) are learned on a fixed set of
# seasons and then frozen, so later games never move them.

library(tidyverse)

add_scenario_inputs <- function(v) {
  v |> mutate(
    run_over_pass = ((fav_off_rush_epa_roll5 - dog_def_rush_epa_roll5) - (dog_off_rush_epa_roll5 - fav_def_rush_epa_roll5)) -
                    ((fav_off_pass_epa_roll5 - dog_def_pass_epa_roll5) - (dog_off_pass_epa_roll5 - fav_def_pass_epa_roll5)),
    epa_delta  = off_epa_play_roll5_diff - def_epa_play_roll5_diff,
    form_delta = (fav_off_epa_play_roll3 - fav_off_epa_play_roll5) - (dog_off_epa_play_roll3 - dog_off_epa_play_roll5))
}

learn_scenario_cuts <- function(v, seasons) {
  d <- add_scenario_inputs(v) |> filter(season %in% seasons, !is.na(fav_cover))
  inj_stats <- map(set_names(c("injury_diff", "fav_n_out", "dog_n_out")),
                   ~ c(mean = mean(d[[.x]], na.rm = TRUE), sd = sd(d[[.x]], na.rm = TRUE)))
  d$inj <- injury_score(d, inj_stats)
  q <- function(x, p) unname(quantile(d[[x]], p, na.rm = TRUE))
  list(seasons = seasons, inj_stats = inj_stats,
       epa_hi = q("epa_delta", 2/3), epa_lo = q("epa_delta", 1/3),
       rop_hi = q("run_over_pass", 2/3), rop_lo = q("run_over_pass", 1/3),
       inj_hi = q("inj", 0.8), inj_lo = q("inj", 0.2),
       form_hi = q("form_delta", 2/3), form_lo = q("form_delta", 1/3),
       to_hi = q("turnover_diff", 2/3), to_lo = q("turnover_diff", 1/3))
}

injury_score <- function(d, s) {
  rowMeans(cbind((d$injury_diff - s$injury_diff[["mean"]]) / s$injury_diff[["sd"]],
                 -(d$fav_n_out - s$fav_n_out[["mean"]]) / s$fav_n_out[["sd"]],
                 (d$dog_n_out - s$dog_n_out[["mean"]]) / s$dog_n_out[["sd"]]), na.rm = TRUE)
}

# Named list of logical vectors, one per condition (favorite's perspective)
scenario_conditions <- function(v, cuts) {
  v <- add_scenario_inputs(v)
  inj <- injury_score(v, cuts$inj_stats)
  yes <- function(x) coalesce(x, FALSE)
  with(v, list(
    spread_le3       = yes(spread_fav <= 3),
    spread_3h_6h     = yes(spread_fav > 3 & spread_fav < 7),
    spread_7_9h      = yes(spread_fav >= 7 & spread_fav < 10),
    spread_10plus    = yes(spread_fav >= 10),
    fav_home         = fav_at_home == "yes",
    fav_road         = fav_at_home != "yes",
    division         = yes(div_game == 1),
    primetime        = yes(is_primetime == 1),
    fav_short_week   = fav_short_week == "yes",
    dog_short_week   = dog_short_week == "yes",
    fav_rest_edge    = yes(rest_diff >= 3),
    dog_rest_edge    = yes(rest_diff <= -3),
    bad_weather      = yes(weather_wind_mph >= 15 | weather_temperature <= 32),
    dome             = yes(is_dome == 1),
    high_total       = yes(over_under_line > 48),
    low_total        = yes(over_under_line < 41),
    early_season     = week <= 4,
    late_season      = week >= 13,
    elo_likes_fav    = yes(elo_spread_residual > 2),
    elo_likes_dog    = yes(elo_spread_residual < -2),
    epa_fav_strong   = yes(epa_delta >= cuts$epa_hi),
    epa_fav_weak     = yes(epa_delta <= cuts$epa_lo),
    fav_run_edge     = yes(run_over_pass >= cuts$rop_hi),
    dog_run_edge     = yes(run_over_pass <= cuts$rop_lo),
    dog_more_hurt    = yes(inj >= cuts$inj_hi),
    fav_more_hurt    = yes(inj <= cuts$inj_lo),
    fav_qb_change    = yes(fav_qb_change_flag == 1),
    dog_qb_change    = yes(dog_qb_change_flag == 1),
    fav_hot          = yes(form_delta >= cuts$form_hi),
    dog_hot          = yes(form_delta <= cuts$form_lo),
    fav_to_lucky     = yes(turnover_diff >= cuts$to_hi),
    dog_to_lucky     = yes(turnover_diff <= cuts$to_lo)
  ))
}

# Does each game match "cond_a & cond_b"?
scenario_fires <- function(conds, scenario) {
  parts <- str_split_1(scenario, " & ")
  reduce(conds[parts], `&`)
}
