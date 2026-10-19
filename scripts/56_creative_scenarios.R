# Creative scenarios — schedule, travel, psychology, public betting
#
# Every scenario below was written down with its reason before testing.
# Built from nflverse schedules (scores, lines, kickoff times, overtime).
# Tested like 54_scenario_search.R (MODE=recent): 2018-2025 win rate,
# seasons above breakeven out of 8, shuffled-outcome luck check, and 2026.

library(tidyverse); library(here); library(nflreadr)
set.seed(42)

norm <- function(x) recode(x, LV = "LVR", LA = "LAR", OAK = "LVR", SD = "LAC", STL = "LAR")
tz_of <- function(raw) case_when(
  raw %in% c("SEA", "SF", "OAK", "LV", "LA", "LAR", "SD", "LAC") ~ -3,
  raw %in% c("DEN", "ARI") ~ -2,
  raw %in% c("CHI", "DAL", "GB", "HOU", "KC", "MIN", "NO", "TEN", "STL") ~ -1,
  TRUE ~ 0)

sched <- load_schedules(2011:2026) |> filter(game_type == "REG") |>
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
    days_rest = as.numeric(difftime(kickoff, lag(kickoff), units = "days")),
    days_to_next = as.numeric(difftime(lead(kickoff), kickoff, units = "days")),
    next_primetime = lead(primetime),
    road_streak = { r <- !home; ave(r, cumsum(!r), FUN = cumsum) }
  ) |>
  group_by(team, season) |>
  mutate(lost_to_opp_earlier = map_lgl(row_number(), function(i)
    any(opp[seq_len(i - 1)] == opp[i] & margin[seq_len(i - 1)] < 0, na.rm = TRUE))) |>
  ungroup()

v <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(!is.na(fav_cover), season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover"),
         fav = if_else(fav_at_home == "yes", str_extract(game_id_norm, "[A-Z]+$"),
                       str_match(game_id_norm, "_\\d+_([A-Z]+)_")[, 2]),
         dog = if_else(fav_at_home == "yes", str_match(game_id_norm, "_\\d+_([A-Z]+)_")[, 2],
                       str_extract(game_id_norm, "[A-Z]+$")))

side_feats <- tg |> select(game_id_norm, team, home, prev_margin, prev_covered, prev_was_fav, prev_ot, prev_mnf,
                           covered_last3, days_rest, days_to_next, next_primetime, road_streak,
                           lost_to_opp_earlier, tz_shift, gametime, weekday, neutral)
g <- v |>
  left_join(side_feats |> rename_with(~ str_c("f_", .x), -c(game_id_norm, team)), by = c("game_id_norm", "fav" = "team")) |>
  left_join(side_feats |> rename_with(~ str_c("d_", .x), -c(game_id_norm, team)), by = c("game_id_norm", "dog" = "team"))

yes <- function(x) coalesce(x, FALSE)
public_teams <- c("DAL", "GB", "PIT", "KC", "SF", "NE")

# name = condition, side = what to bet, why = the reason (written before testing)
scenarios <- list(
  list("fav_covered_3_straight", "underdog", "Market inflates lines on teams on an ATS heater",
       with(g, yes(f_covered_last3))),
  list("fav_upset_last_week", "favorite", "Market overreacts to a favorite's outright loss",
       with(g, yes(f_prev_was_fav & f_prev_margin < 0))),
  list("fav_off_blowout_win", "underdog", "Letdown + inflated line after a 21+ point win",
       with(g, yes(f_prev_margin >= 21))),
  list("dog_off_blowout_loss", "underdog", "Public overreacts to an ugly loss; dog undervalued",
       with(g, yes(d_prev_margin <= -21))),
  list("fav_sandwich_primetime_next", "underdog", "Lookahead: favorite has a primetime game next",
       with(g, yes(f_next_primetime & !f_home))),
  list("fav_bye_next", "underdog", "Lookahead to the bye", with(g, yes(f_days_to_next >= 13))),
  list("dog_off_bye_fav_not", "underdog", "Extra prep week for the underdog",
       with(g, yes(d_days_rest >= 13 & f_days_rest < 13))),
  list("west_fav_1pm_east", "underdog", "Body clock: West Coast favorite at 1pm Eastern",
       with(g, yes(!f_home & f_tz_shift >= 2 & str_starts(f_gametime, "13")))),
  list("west_dog_1pm_east", "favorite", "Body clock: West Coast underdog at 1pm Eastern",
       with(g, yes(!d_home & d_tz_shift >= 2 & str_starts(d_gametime, "13")))),
  list("road_fav_long_trip", "underdog", "Road favorite crossing 2+ time zones",
       with(g, yes(!f_home & abs(f_tz_shift) >= 2))),
  list("fav_third_straight_road", "underdog", "Road fatigue on the third straight road game",
       with(g, yes(f_road_streak >= 3))),
  list("dog_revenge", "underdog", "Underdog lost to this opponent earlier this season",
       with(g, yes(d_lost_to_opp_earlier))),
  list("fav_after_overtime", "underdog", "Fatigue after an overtime game",
       with(g, yes(f_prev_ot))),
  list("road_fav_after_mnf", "underdog", "Short prep after Monday night, then travel",
       with(g, yes(f_prev_mnf & !f_home))),
  list("public_team_fav", "underdog", "Popular teams draw public money, inflating their lines",
       with(g, fav %in% public_teams)),
  list("home_dog_division", "underdog", "Division familiarity + home crowd for the underdog",
       with(g, yes(fav_at_home != "yes" & div_game == 1))),
  list("thursday_home_dog", "underdog", "Short-week chaos favors the home underdog",
       with(g, yes(fav_at_home != "yes" & f_weekday == "Thursday"))),
  list("dog_at_denver_altitude", "underdog", "Visiting favorites at altitude",
       with(g, dog == "DEN" & fav_at_home != "yes")),
  list("double_digit_dog_off_cover", "favorite", "Big dog that just covered gets overbet next week",
       with(g, yes(spread_fav >= 10 & d_prev_covered)))
)

evaluate <- function(y) map_dfr(scenarios, function(s) {
  hit <- s[[4]]; win <- if (s[[2]] == "favorite") y == 1 else y == 0
  per <- map_dfr(2018:2025, function(yr) {
    m <- hit & g$season == yr
    tibble(n = sum(m), w = sum(m & win))
  })
  rec <- hit & g$season %in% 2018:2025; old <- hit & g$season %in% 2012:2017; new <- hit & g$season == 2026
  tibble(scenario = s[[1]], bet = s[[2]], why = s[[3]],
         per_season = sum(rec) / 8,
         win_2018_25 = sum(rec & win) / sum(rec),
         seasons_won = sum(per$w / pmax(per$n, 1) > 0.524 & per$n > 0),
         win_2012_17 = sum(old & win) / sum(old),
         fresh_w = sum(new & win), fresh_n = sum(new))
})

res <- evaluate(g$y)
pass <- function(r) r |> filter(per_season >= 5, win_2018_25 >= 0.55, seasons_won >= 6)
shuffled <- map_dbl(1:200, function(i) {
  ys <- g |> group_by(season) |> mutate(ys = sample(y)) |> pull(ys)
  nrow(pass(evaluate(ys)))
})

cat(sprintf("%d creative scenarios tested (reasons written first)\n", length(scenarios)))
cat(sprintf("Pass (5+/season, >=55%% in 2018-25, above breakeven in 6+ of 8 seasons): %d | luck alone: %.1f on average\n\n",
            nrow(pass(res)), mean(shuffled)))
res |> arrange(desc(win_2018_25)) |>
  transmute(scenario, bet, per_season = round(per_season, 1),
            `2018-25` = sprintf("%.1f%%", 100 * win_2018_25), seasons_won = str_c(seasons_won, "/8"),
            `2012-17` = sprintf("%.1f%%", 100 * win_2012_17),
            `2026` = sprintf("%d-%d", fresh_w, fresh_n - fresh_w), why) |>
  print(n = Inf, width = Inf)
write_csv(res, here("results", "creative_scenarios_v37.csv"))
