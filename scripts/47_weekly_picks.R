# Weekly picks for the current season
#
# Weekly routine (Tuesday after Monday Night Football, then again before
# kickoff if lines or injuries move):
#   Rscript scripts/02_prepare_recent_data.R      # pull latest data + lines
#   Rscript scripts/01_build_model_data.R
#   Rscript scripts/10_build_advanced_features.R
#   PYTHONPATH=. /usr/bin/python3 scripts/run_pipeline.py --phase features && ... assemble
#   Rscript scripts/19c_rebuild_v3_clean.R
#   Rscript scripts/20_finalize_modeling_table.R
#   Rscript scripts/47_weekly_picks.R
#
# Models are trained once per season on every completed game before it
# (the same rule the 2023-2025 backtest used), then scored on upcoming games.
# Stake: flat 1% of bankroll per bet (see 45_calibrated_kelly.R).

library(tidyverse); library(here)
source(here("scripts", "model_functions.R"))

v36 <- read_rds(here("objects", "modeling_data_v36.rds"))
upcoming <- read_rds(here("objects", "upcoming_games.rds"))
calibrators <- read_rds(here("objects", "calibrators_v36.rds"))

season_now <- max(upcoming$season)
sf <- stable_features(v36)
hist <- v36 |> filter(!is.na(fav_cover), season < season_now)
games <- v36 |> filter(is_upcoming)

p_cover <- cover_members |>
  map(~ fit_predict(hist |> filter(season >= .x[1]), games, sf, .x[2], .x[3], TRUE)) |>
  reduce(`+`) / length(cover_members)
p_nocover <- fit_predict(hist |> filter(season >= nocover_member[1]), games, sf,
                         nocover_member[2], nocover_member[3], FALSE)

calibrate <- function(coefs, p) plogis(coefs[1] + coefs[2] * qlogis(pmin(pmax(p, 0.01), 0.99)))

picks <- games |>
  select(game_id_norm, week, spread_fav) |>
  mutate(p_cover = p_cover, p_nocover = p_nocover) |>
  left_join(upcoming |> select(game_id_norm, gameday, gametime, home_team, away_team, spread_line),
            by = "game_id_norm") |>
  mutate(
    # nflverse spread_line is from the home side: positive = home favored
    favorite = if_else(spread_line >= 0, home_team, away_team),
    underdog = if_else(spread_line >= 0, away_team, home_team),
    bet_type = case_when(
      p_cover > cover_threshold & !(p_nocover > nocover_threshold & p_nocover > p_cover) ~ "cover",
      p_nocover > nocover_threshold ~ "no_cover",
      TRUE ~ "pass"),
    pick = case_when(
      bet_type == "cover"    ~ str_glue("{favorite} -{spread_fav}"),
      bet_type == "no_cover" ~ str_glue("{underdog} +{spread_fav}"),
      TRUE ~ "no bet"),
    model_p = case_when(bet_type == "cover" ~ p_cover, bet_type == "no_cover" ~ p_nocover),
    calibrated_p = case_when(
      bet_type == "cover"    ~ calibrate(calibrators$cover, p_cover),
      bet_type == "no_cover" ~ calibrate(calibrators$nocover, p_nocover)),
    stake = if_else(bet_type == "pass", "", "1% of bankroll"),
    matchup = str_glue("{away_team} @ {home_team}")
  ) |>
  # Drop games that have already kicked off (schedule times are US Eastern)
  filter(as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York") > Sys.time()) |>
  arrange(gameday, gametime) |>
  select(week, gameday, gametime, matchup, line = spread_fav, favorite, pick, stake,
         model_p, calibrated_p, p_cover, p_nocover)

out <- here("results", str_glue("picks_{season_now}_wk{str_pad(max(picks$week), 2, pad = '0')}.csv"))
write_csv(picks, out)

cat(sprintf("\n=== %d PICKS — weeks %s ===\n", season_now, paste(unique(picks$week), collapse = ", ")))
cat("Lines from nflverse at run time; bet only if your book's line is the same or better.\n\n")
picks |>
  mutate(across(c(model_p, calibrated_p, p_cover, p_nocover), ~ round(.x, 3))) |>
  print(n = Inf, width = Inf)
cat(sprintf("\n%d bets of %d games. Saved %s\n", sum(picks$pick != "no bet"), nrow(picks), out))
