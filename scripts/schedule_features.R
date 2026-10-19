# Schedule / situational features shared by 56, 57 and the trackers.
# One row per team-game in date order, then attached to each game from the
# favorite's (f_) and underdog's (d_) side. Uses only games before each one.

library(tidyverse); library(nflreadr)

add_schedule_features <- function(v, seasons = 2011:2026) {
  norm <- function(x) recode(x, LV = "LVR", LA = "LAR", OAK = "LVR", SD = "LAC", STL = "LAR")
  tz_of <- function(raw) case_when(
    raw %in% c("SEA", "SF", "OAK", "LV", "LA", "LAR", "SD", "LAC") ~ -3,
    raw %in% c("DEN", "ARI") ~ -2,
    raw %in% c("CHI", "DAL", "GB", "HOU", "KC", "MIN", "NO", "TEN", "STL") ~ -1,
    TRUE ~ 0)

  sched <- load_schedules(seasons) |> filter(game_type == "REG") |>
    mutate(kickoff = as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York"),
           game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", norm(away_team), "_", norm(home_team)),
           neutral = location == "Neutral", site_tz = if_else(neutral, NA_real_, tz_of(home_team)))

  # One row per team per game, in date order
  tg <- bind_rows(
    sched |> transmute(game_id_norm, season, week, kickoff, gametime, weekday, overtime, neutral, site_tz,
                       raw = home_team, team = norm(home_team), opp = norm(away_team), home = TRUE,
                       pts = home_score, opp_pts = away_score, line_for = spread_line),
    sched |> transmute(game_id_norm, season, week, kickoff, gametime, weekday, overtime, neutral, site_tz,
                       raw = away_team, team = norm(away_team), opp = norm(home_team), home = FALSE,
                       pts = away_score, opp_pts = home_score, line_for = -spread_line)) |>
    # line_for: points this team is favored by (negative = underdog)
    mutate(margin = pts - opp_pts, ats = margin - line_for,
           covered = ats > 0, was_fav = line_for > 0,
           primetime = weekday %in% c("Monday", "Thursday") | (weekday == "Sunday" & gametime >= "19:00"),
           tz_shift = site_tz - tz_of(raw)) |>
    arrange(team, kickoff) |>
    group_by(team) |>
    mutate(
      prev_margin = lag(margin), prev_covered = lag(covered), prev_was_fav = lag(was_fav),
      prev_ot = coalesce(lag(overtime) == 1, FALSE), prev_mnf = coalesce(lag(weekday) == "Monday", FALSE),
      covered_last3 = lag(covered) & lag(covered, 2) & lag(covered, 3),
      no_cover_last3 = !lag(covered) & !lag(covered, 2) & !lag(covered, 3),
      days_rest = as.numeric(difftime(kickoff, lag(kickoff), units = "days")),
      days_to_next = as.numeric(difftime(lead(kickoff), kickoff, units = "days")),
      next_primetime = lead(primetime),
      road_streak = { r <- !home; ave(r, cumsum(!r), FUN = cumsum) }
    ) |>
    group_by(team, season) |>
    mutate(lost_to_opp_earlier = map_lgl(row_number(), function(i)
      any(opp[seq_len(i - 1)] == opp[i] & margin[seq_len(i - 1)] < 0, na.rm = TRUE))) |>
    ungroup()

  v <- v |>
    mutate(fav = if_else(fav_at_home == "yes", str_extract(game_id_norm, "[A-Z]+$"),
                         str_match(game_id_norm, "_\\d+_([A-Z]+)_")[, 2]),
           dog = if_else(fav_at_home == "yes", str_match(game_id_norm, "_\\d+_([A-Z]+)_")[, 2],
                         str_extract(game_id_norm, "[A-Z]+$")))

  side_feats <- tg |> select(game_id_norm, team, home, prev_margin, prev_covered, prev_was_fav, prev_ot, prev_mnf,
                             covered_last3, no_cover_last3, days_rest, days_to_next, next_primetime, road_streak,
                             lost_to_opp_earlier, tz_shift, gametime, weekday, neutral)
  g <- v |>
    left_join(side_feats |> rename_with(~ str_c("f_", .x), -c(game_id_norm, team)), by = c("game_id_norm", "fav" = "team")) |>
    left_join(side_feats |> rename_with(~ str_c("d_", .x), -c(game_id_norm, team)), by = c("game_id_norm", "dog" = "team"))


  g
}

# Situation x matchup conditions used by 57_combo_scenarios.R and the tracker.
# Cutoffs are learned once on `seasons` and frozen.
learn_combo_cuts <- function(g, seasons) {
  d <- g |> filter(season %in% seasons)
  q <- function(x) unname(quantile(x, 2/3, na.rm = TRUE))
  list(stat = learn_scenario_cuts(g, seasons),
       fav_rush_hi = q(d$fav_off_rush_epa_roll5 - d$dog_def_rush_epa_roll5),
       dog_rush_hi = q(d$dog_off_rush_epa_roll5 - d$fav_def_rush_epa_roll5))
}

combo_conditions <- function(g, cuts) {
  yes <- function(x) coalesce(x, FALSE)
  c(scenario_conditions(g, cuts$stat), with(g, list(
    fav_rush_mismatch  = yes(fav_off_rush_epa_roll5 - dog_def_rush_epa_roll5 >= cuts$fav_rush_hi),
    dog_rush_mismatch  = yes(dog_off_rush_epa_roll5 - fav_def_rush_epa_roll5 >= cuts$dog_rush_hi),
    fav_no_cover_last3 = yes(f_no_cover_last3),
    fav_covered_last3  = yes(f_covered_last3),
    dog_no_cover_last3 = yes(d_no_cover_last3),
    dog_covered_last3  = yes(d_covered_last3),
    fav_lost_last      = yes(f_prev_margin < 0),
    fav_upset_last     = yes(f_prev_was_fav & f_prev_margin < 0),
    dog_won_last       = yes(d_prev_margin > 0),
    dog_upset_win_last = yes(!d_prev_was_fav & d_prev_margin > 0),
    fav_blowout_win    = yes(f_prev_margin >= 21),
    dog_blowout_loss   = yes(d_prev_margin <= -21),
    fav_off_bye        = yes(f_days_rest >= 13),
    dog_off_bye        = yes(d_days_rest >= 13),
    fav_short_rest     = yes(f_days_rest <= 6))))
}
