# Creative scenarios — schedule, travel, psychology, public betting
#
# Every scenario below was written down with its reason before testing.
# Built from nflverse schedules (scores, lines, kickoff times, overtime).
# Tested like 54_scenario_search.R (MODE=recent): 2018-2025 win rate,
# seasons above breakeven out of 8, shuffled-outcome luck check, and 2026.

library(tidyverse); library(here); library(nflreadr)
set.seed(42)

source(here("scripts", "schedule_features.R"))
g <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(!is.na(fav_cover), season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover")) |>
  add_schedule_features()

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
