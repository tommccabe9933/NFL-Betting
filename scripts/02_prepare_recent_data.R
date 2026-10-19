# Prepare 2025-2026 inputs for the v3.6 rebuild
#
# The original pipeline stops at 2024 in two places: the Kaggle team-stat CSVs
# and the Python raw downloads (no player stats after 2024, nothing for 2026).
# This script fills those gaps so 01 / 10 / the Python store / 19c can run
# through the current season:
#   1. nflverse raw parquet for 2025-2026 in the Python pipeline's file layout
#   2. Kaggle-format team stats rebuilt from play-by-play (2025+)
#   3. Spreadspoke-format game rows for 2026 (lines + weather)
#   4. The list of upcoming games to predict

library(tidyverse); library(here); library(nflreadr); library(arrow); library(httr2)
source(here("scripts", "00_project_setup.R"))

recent_seasons <- 2025:2026
current_season <- 2026

# ================================================================
# 1. Raw nflverse files for the Python pipeline
# ================================================================
raw <- function(...) here("data", "raw", ...)
pbp_keep_cols <- c("game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
  "down", "ydstogo", "yardline_100", "play_type", "pass", "rush", "epa", "cpoe", "air_yards",
  "yards_after_catch", "qb_hit", "sack", "interception", "fumble_lost", "touchdown", "first_down",
  "success", "shotgun", "no_huddle", "qb_dropback", "home_team", "away_team", "game_date",
  "passer_player_id", "rusher_player_id", "receiver_player_id", "passer_player_name",
  "rusher_player_name", "nflverse_game_id")
save_pq <- function(df, path) if (!is.null(df) && nrow(df) > 0) write_parquet(df, path)
try_load <- function(expr) tryCatch(expr, error = function(e) NULL)

for (s in recent_seasons) {
  if (s == current_season || !file.exists(raw("pbp", str_glue("pbp_{s}.parquet")))) {
    # Same column subset the Python ingest keeps (src/config.py PBP_KEEP_COLS)
    save_pq(load_pbp(s) |> select(any_of(pbp_keep_cols)), raw("pbp", str_glue("pbp_{s}.parquet")))
  }
  save_pq(load_schedules(s), raw("schedules", str_glue("schedules_{s}.parquet")))
  save_pq(try_load(load_injuries(s)), raw("injuries", str_glue("injuries_{s}.parquet")))
  save_pq(try_load(load_ftn_charting(s)), raw("ftn", str_glue("ftn_charting_{s}.parquet")))
  save_pq(try_load(load_depth_charts(s)), raw("depth_charts", str_glue("depth_charts_{s}.parquet")))
  for (st in c("passing", "rushing", "receiving")) {
    save_pq(try_load(load_nextgen_stats(s, st)), raw("ngs", str_glue("ngs_{st}_{s}.parquet")))
  }
  for (st in c("pass", "rush", "rec", "def")) {
    save_pq(try_load(load_pfr_advstats(s, st, "week")), raw("pfr", str_glue("advstats_week_{st}_{s}.parquet")))
  }
  # nflverse renamed recent_team -> team in the 2025 player-stats release
  ps <- try_load(load_player_stats(s))
  if (!is.null(ps)) {
    if (!"recent_team" %in% names(ps)) ps <- ps |> rename(recent_team = team)
    save_pq(ps, raw("player_stats", str_glue("player_stats_{s}.parquet")))
  }
}

# ================================================================
# 2. Upcoming games: unplayed regular-season games in the next week
# ================================================================
sched <- load_schedules(current_season) |>
  filter(game_type == "REG") |>
  mutate(home_team = normalize_team(home_team), away_team = normalize_team(away_team),
         game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", away_team, "_", home_team))

next_week <- min(sched$week[is.na(sched$result)])
upcoming <- sched |>
  filter(is.na(result), week <= next_week + 1, !is.na(spread_line)) |>
  select(game_id_norm, season, week, gameday, gametime, home_team, away_team, stadium, roof, spread_line, total_line)

write_rds(upcoming, here("objects", "upcoming_games.rds"))
write_parquet(
  bind_rows(upcoming |> transmute(season, week, team = home_team, game_id_norm),
            upcoming |> transmute(season, week, team = away_team, game_id_norm)),
  here("data", "processed", "normalized", "upcoming_team_weeks.parquet"))
cat(sprintf("Upcoming: %d games (weeks %s)\n", nrow(upcoming),
            paste(sort(unique(upcoming$week)), collapse = ", ")))

# ================================================================
# 3. Kaggle-format team stats from play-by-play (2025+)
# Column definitions checked against the Kaggle file for 2024.
# ================================================================
# Full nflverse pbp (the Python pipeline's raw files keep only a column subset)
pbp_recent <- load_pbp(recent_seasons) |>
  filter(season_type == "REG")

off_stats <- pbp_recent |>
  filter(!is.na(posteam), posteam != "") |>
  group_by(game_id, season, week, team = posteam) |>
  summarise(
    pass_attempts = sum(complete_pass + incomplete_pass, na.rm = TRUE),
    complete_pass = sum(complete_pass, na.rm = TRUE),
    passing_yards = sum(passing_yards, na.rm = TRUE),
    rush_attempts = sum(rush_attempt, na.rm = TRUE),
    rushing_yards = sum(rushing_yards, na.rm = TRUE),
    third_down_converted = sum(third_down_converted, na.rm = TRUE),
    third_down_failed = sum(third_down_failed, na.rm = TRUE),
    interception = sum(interception, na.rm = TRUE),
    fumble_lost = sum(fumble_lost, na.rm = TRUE),
    fumble_forced = sum(fumble_forced, na.rm = TRUE),
    offense_snaps = sum(play_type %in% c("pass", "run") | qb_kneel == 1 | qb_spike == 1, na.rm = TRUE),
    total_off_points = sum(6 * (touchdown == 1 & td_team == posteam) +
                           3 * (field_goal_result %in% "made") +
                           (extra_point_result %in% "good") +
                           2 * (two_point_conv_result %in% "success"), na.rm = TRUE),
    .groups = "drop") |>
  mutate(total_off_yards = passing_yards + rushing_yards, season_type = "REG")

def_stats <- pbp_recent |>
  filter(!is.na(defteam), defteam != "") |>
  group_by(game_id, season, week, team = defteam) |>
  summarise(interception = sum(interception, na.rm = TRUE),
            fumble_forced = sum(fumble_forced, na.rm = TRUE),
            fumble_lost = sum(fumble_lost, na.rm = TRUE),
            sack = sum(sack, na.rm = TRUE),
            defense_snaps = sum(play_type %in% c("pass", "run") | qb_kneel == 1 | qb_spike == 1, na.rm = TRUE),
            total_off_points = 0, .groups = "drop") |>
  mutate(season_type = "REG")

# Upcoming games get empty rows so the lagged rolling means land on them
upcoming_rows <- bind_rows(
  upcoming |> transmute(game_id = game_id_norm, season, week, team = home_team),
  upcoming |> transmute(game_id = game_id_norm, season, week, team = away_team)) |>
  mutate(season_type = "REG")

write_csv(bind_rows(off_stats, upcoming_rows), here("data", "weekly_team_stats_offense_recent.csv"))
write_csv(bind_rows(def_stats, upcoming_rows), here("data", "weekly_team_stats_defense_recent.csv"))

# ================================================================
# 4. Spreadspoke-format rows for the current season
# Domes / closed / undecided retractable roofs = 72F, 0 mph (Spreadspoke
# convention). Outdoor games use nflverse weather once played and an
# Open-Meteo forecast at kickoff before that.
# ================================================================
team_names <- read_csv(here("data", "nfl_all_seasons.csv"), show_col_types = FALSE) |>
  filter(schedule_season == 2025) |>
  distinct(team_name = team_home) |>
  left_join(team_lookup, by = "team_name")

stadium_coords <- tribble(
  ~stadium, ~lat, ~lon,
  "Lumen Field", 47.595, -122.332, "Bank of America Stadium", 35.226, -80.853,
  "Paycor Stadium", 39.095, -84.516, "EverBank Stadium", 30.324, -81.637,
  "Acrisure Stadium", 40.447, -80.016, "Nissan Stadium", 36.166, -86.771,
  "Lincoln Financial Field", 39.901, -75.168, "MetLife Stadium", 40.814, -74.074,
  "GEHA Field at Arrowhead Stadium", 39.049, -94.484, "Highmark Stadium", 42.774, -78.787,
  "M&T Bank Stadium", 39.278, -76.623, "Soldier Field", 41.862, -87.617,
  "Gillette Stadium", 42.091, -71.264, "Raymond James Stadium", 27.976, -82.503,
  "Empower Field at Mile High", 39.744, -105.020, "Levi's Stadium", 37.403, -121.970,
  "Lambeau Field", 44.501, -88.062, "Huntington Bank Field", 41.506, -81.700,
  "Hard Rock Stadium", 25.958, -80.239, "Northwest Stadium", 38.908, -76.865,
  "SoFi Stadium", 33.953, -118.339, "Tottenham Hotspur Stadium", 51.604, -0.066,
  "Wembley Stadium", 51.556, -0.280, "Stade de France", 48.924, 2.360,
  "Bernabeu", 40.453, -3.688, "FC Bayern Munich Stadium", 48.219, 11.625,
  "Estadio Banorte", 19.303, -99.150, "Maracana Stadium", -22.912, -43.230,
  "Melbourne Cricket Ground", -37.820, 144.983
)

forecast_weather <- function(lat, lon, gameday, gametime) {
  resp <- request("https://api.open-meteo.com/v1/forecast") |>
    req_url_query(latitude = lat, longitude = lon, hourly = "temperature_2m,wind_speed_10m",
                  temperature_unit = "fahrenheit", wind_speed_unit = "mph",
                  timezone = "America/New_York", start_date = gameday, end_date = gameday) |>
    req_perform() |> resp_body_json()
  hour <- as.integer(str_sub(gametime, 1, 2))
  idx <- min(hour + 1, length(resp$hourly$time))
  c(temp = round(resp$hourly$temperature_2m[[idx]]), wind = round(resp$hourly$wind_speed_10m[[idx]]))
}

cur <- sched |>
  left_join(stadium_coords, by = "stadium") |>
  mutate(indoor = roof %in% c("dome", "closed") | (is.na(roof) & home_team %in% c("ATL", "IND", "HOU", "ARI", "DAL", "DET", "MIN", "NO", "LVR")))

wx <- pmap_dfr(cur |> select(game_id_norm, result, indoor, temp, wind, lat, lon, gameday, gametime),
  function(game_id_norm, result, indoor, temp, wind, lat, lon, gameday, gametime) {
    if (indoor) return(tibble(game_id_norm, weather_temperature = 72, weather_wind_mph = 0))
    if (!is.na(result)) return(tibble(game_id_norm, weather_temperature = temp, weather_wind_mph = wind))
    if (!(game_id_norm %in% upcoming$game_id_norm) || is.na(lat))
      return(tibble(game_id_norm, weather_temperature = NA_real_, weather_wind_mph = NA_real_))
    f <- tryCatch(forecast_weather(lat, lon, gameday, gametime), error = function(e) c(temp = NA, wind = NA))
    tibble(game_id_norm, weather_temperature = f[["temp"]], weather_wind_mph = f[["wind"]])
  })

games_cur <- cur |>
  left_join(wx, by = "game_id_norm") |>
  left_join(team_names |> rename(team_home = team_name), by = c("home_team" = "team")) |>
  left_join(team_names |> rename(team_away = team_name), by = c("away_team" = "team")) |>
  transmute(
    schedule_date = format(as.Date(gameday), "%m/%d/%Y"),
    schedule_season = season, schedule_week = as.character(week), schedule_playoff = FALSE,
    team_home, score_home = home_score, score_away = away_score, team_away,
    # nflverse spread_line is from the home side: positive = home favored
    team_favorite_id = case_when(spread_line > 0 ~ home_team, spread_line < 0 ~ away_team, TRUE ~ home_team),
    spread_favorite = -abs(spread_line),
    over_under_line = total_line, stadium,
    stadium_neutral = location == "Neutral",
    weather_temperature, weather_wind_mph, weather_humidity = NA_real_, weather_detail = NA_character_
  )

write_csv(games_cur, here("data", str_glue("nfl_games_{current_season}.csv")))
cat(sprintf("Wrote %d %d games (%d played, %d upcoming with forecast)\n", nrow(games_cur), current_season,
            sum(!is.na(games_cur$score_home)), sum(!is.na(wx$weather_temperature[wx$game_id_norm %in% upcoming$game_id_norm]))))
