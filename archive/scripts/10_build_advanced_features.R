# Build advanced features from nflverse data sources
# Pulls: NGS, FTN charting, PFR advanced stats, injuries, schedules, player stats
# Outputs team-week level features merged into modeling data

library(tidyverse)
library(tidymodels)
library(here)
library(nflreadr)
library(slider)

source(here("scripts", "00_project_setup.R"))

cat("Loading base modeling data...\n")
model_data <- read_rds(here("objects", "modeling_data.rds"))
team_game_full <- read_rds(here("objects", "team_game_full.rds"))

# Load PBP early — needed by FTN join and situational EPA
cat("Loading play-by-play data (this takes a minute)...\n")
pbp_seasons <- 2012:2025
pbp_raw <- map(pbp_seasons, ~ {
  tryCatch(load_pbp(.x), error = function(e) NULL)
}) |> compact() |> bind_rows()

pbp_plays <- pbp_raw |>
  filter(season_type == "REG", rush == 1 | pass == 1,
         !is.na(epa), !is.na(posteam), posteam != "")
cat("  PBP plays loaded:", nrow(pbp_plays), "\n")

# We need team-week mapping from team_game_full
game_team_map <- team_game_full |>
  select(game_id_norm, season, week, team_norm, team_favorite_id, home_away) |>
  mutate(is_fav = team_norm == team_favorite_id)

# ============================================================
# 1. SCHEDULE FEATURES (div_game, roof, surface, QBs, coaches)
# ============================================================
cat("Loading schedule data...\n")
schedule_seasons <- 2012:2025
schedules <- load_schedules(seasons = schedule_seasons)

schedule_features <- schedules |>
  filter(game_type == "REG") |>
  mutate(
    away_team = normalize_team(away_team),
    home_team = normalize_team(home_team),
    game_id_norm = str_c(
      season, "_",
      str_pad(week, 2, pad = "0"), "_",
      away_team, "_", home_team
    ),
    is_primetime = weekday %in% c("Monday", "Thursday") |
      (weekday == "Sunday" & as.numeric(str_extract(gametime, "^\\d+")) >= 7),
    is_dome = roof %in% c("dome", "closed"),
    is_turf = surface != "grass"
  ) |>
  select(game_id_norm, season, week, div_game, is_primetime, is_dome, is_turf,
         away_qb_name, home_qb_name, away_coach, home_coach, referee,
         home_rest, away_rest)

cat("  Schedule features:", nrow(schedule_features), "games\n")

# ============================================================
# 2. QB-LEVEL FEATURES (from player stats)
# ============================================================
cat("Loading player stats...\n")
player_stats_seasons <- 2012:2025
player_stats <- map(player_stats_seasons, ~ {
  tryCatch(load_player_stats(seasons = .x), error = function(e) NULL)
}) |> compact() |> bind_rows()

# Get QB stats per team per week
qb_weekly <- player_stats |>
  filter(position == "QB", season_type == "REG") |>
  mutate(team = normalize_team(team)) |>
  group_by(season, week, team) |>
  summarize(
    qb_passing_epa = sum(passing_epa, na.rm = TRUE),
    qb_cpoe = weighted.mean(passing_cpoe, attempts, na.rm = TRUE),
    qb_attempts = sum(attempts, na.rm = TRUE),
    qb_sacks = sum(sacks_suffered, na.rm = TRUE),
    qb_air_yards = sum(passing_air_yards, na.rm = TRUE),
    qb_yac = sum(passing_yards_after_catch, na.rm = TRUE),
    qb_passing_first_downs = sum(passing_first_downs, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    qb_epa_per_play = qb_passing_epa / pmax(qb_attempts, 1),
    qb_sack_rate = qb_sacks / pmax(qb_attempts + qb_sacks, 1),
    qb_air_yards_per_att = qb_air_yards / pmax(qb_attempts, 1),
    qb_yac_per_comp = qb_yac / pmax(qb_attempts * 0.65, 1)  # rough comp rate
  )

# Rolling QB features (lagged)
qb_rolling <- qb_weekly |>
  group_by(team) |>
  arrange(season, week, .by_group = TRUE) |>
  mutate(
    qb_roll5_epa = lag(slide_dbl(qb_epa_per_play, mean, .before = 4, .complete = FALSE)),
    qb_roll5_cpoe = lag(slide_dbl(qb_cpoe, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    qb_roll5_sack_rate = lag(slide_dbl(qb_sack_rate, mean, .before = 4, .complete = FALSE)),
    qb_roll5_air_yards = lag(slide_dbl(qb_air_yards_per_att, mean, .before = 4, .complete = FALSE))
  ) |>
  ungroup() |>
  select(season, week, team, starts_with("qb_roll5_"))

cat("  QB rolling features:", nrow(qb_rolling), "team-weeks\n")

# ============================================================
# 3. NEXT GEN STATS (NGS)
# ============================================================
cat("Loading Next Gen Stats...\n")
ngs_seasons <- 2021:2025  # NGS only available from 2016+

ngs_pass <- map(ngs_seasons, ~ {
  tryCatch(load_nextgen_stats(seasons = .x, stat_type = "passing"), error = function(e) NULL)
}) |> compact() |> bind_rows()

ngs_rush <- map(ngs_seasons, ~ {
  tryCatch(load_nextgen_stats(seasons = .x, stat_type = "rushing"), error = function(e) NULL)
}) |> compact() |> bind_rows()

# Aggregate NGS passing to team-week level
ngs_pass_team <- ngs_pass |>
  filter(week > 0) |>  # week 0 = season aggregate
  mutate(team_abbr = normalize_team(team_abbr)) |>
  group_by(season, week, team_abbr) |>
  summarize(
    ngs_time_to_throw = weighted.mean(avg_time_to_throw, attempts, na.rm = TRUE),
    ngs_completed_air_yards = weighted.mean(avg_completed_air_yards, attempts, na.rm = TRUE),
    ngs_aggressiveness = weighted.mean(aggressiveness, attempts, na.rm = TRUE),
    ngs_cpoe = weighted.mean(completion_percentage_above_expectation, attempts, na.rm = TRUE),
    .groups = "drop"
  )

# Aggregate NGS rushing to team-week level
ngs_rush_team <- ngs_rush |>
  filter(week > 0) |>
  mutate(team_abbr = normalize_team(team_abbr)) |>
  group_by(season, week, team_abbr) |>
  summarize(
    ngs_rush_yoe = weighted.mean(rush_yards_over_expected_per_att, rush_attempts, na.rm = TRUE),
    ngs_rush_efficiency = weighted.mean(efficiency, rush_attempts, na.rm = TRUE),
    ngs_pct_8plus_defenders = weighted.mean(percent_attempts_gte_eight_defenders, rush_attempts, na.rm = TRUE),
    .groups = "drop"
  )

# Rolling NGS features
ngs_team <- ngs_pass_team |>
  left_join(ngs_rush_team, by = c("season", "week", "team_abbr")) |>
  group_by(team_abbr) |>
  arrange(season, week, .by_group = TRUE) |>
  mutate(
    ngs_roll5_ttt = lag(slide_dbl(ngs_time_to_throw, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    ngs_roll5_cpoe = lag(slide_dbl(ngs_cpoe, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    ngs_roll5_aggressiveness = lag(slide_dbl(ngs_aggressiveness, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    ngs_roll5_rush_yoe = lag(slide_dbl(ngs_rush_yoe, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    ngs_roll5_rush_eff = lag(slide_dbl(ngs_rush_efficiency, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
  ) |>
  ungroup() |>
  select(season, week, team_abbr, starts_with("ngs_roll5_"))

cat("  NGS rolling features:", nrow(ngs_team), "team-weeks\n")

# ============================================================
# 4. FTN CHARTING (play action, RPO, blitz, pressure)
# ============================================================
cat("Loading FTN charting data...\n")
ftn_seasons <- 2022:2025  # FTN typically available from 2022

ftn <- map(ftn_seasons, ~ {
  tryCatch(load_ftn_charting(seasons = .x), error = function(e) NULL)
}) |> compact() |> bind_rows()

if (nrow(ftn) > 0) {
  # FTN doesn't have team columns — join with PBP to get posteam/defteam
  pbp_team_lookup <- pbp_plays |>
    select(season, week, play_id, posteam, defteam) |>
    distinct()

  ftn_with_teams <- ftn |>
    left_join(pbp_team_lookup,
              by = c("season", "week", "nflverse_play_id" = "play_id")) |>
    filter(!is.na(posteam))

  # Aggregate to team-week (offensive perspective)
  ftn_offense <- ftn_with_teams |>
    mutate(posteam = normalize_team(posteam)) |>
    group_by(season, week, posteam) |>
    summarize(
      ftn_play_action_rate = mean(is_play_action, na.rm = TRUE),
      ftn_screen_rate = mean(is_screen_pass, na.rm = TRUE),
      ftn_rpo_rate = mean(is_rpo, na.rm = TRUE),
      ftn_n_plays = n(),
      .groups = "drop"
    )

  # Defensive perspective (how many blitzers, pass rushers faced)
  ftn_defense <- ftn_with_teams |>
    mutate(defteam = normalize_team(defteam)) |>
    group_by(season, week, defteam) |>
    summarize(
      ftn_avg_blitzers = mean(n_blitzers, na.rm = TRUE),
      ftn_avg_pass_rushers = mean(n_pass_rushers, na.rm = TRUE),
      ftn_pressure_worthy = mean(is_qb_fault_sack, na.rm = TRUE),
      .groups = "drop"
    )

  # Rolling FTN features
  ftn_off_rolling <- ftn_offense |>
    group_by(posteam) |>
    arrange(season, week, .by_group = TRUE) |>
    mutate(
      ftn_roll5_play_action = lag(slide_dbl(ftn_play_action_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
      ftn_roll5_rpo = lag(slide_dbl(ftn_rpo_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
    ) |>
    ungroup() |>
    select(season, week, posteam, starts_with("ftn_roll5_"))

  ftn_def_rolling <- ftn_defense |>
    group_by(defteam) |>
    arrange(season, week, .by_group = TRUE) |>
    mutate(
      ftn_roll5_blitz = lag(slide_dbl(ftn_avg_blitzers, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
      ftn_roll5_pass_rush = lag(slide_dbl(ftn_avg_pass_rushers, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
    ) |>
    ungroup() |>
    select(season, week, defteam, starts_with("ftn_roll5_"))

  cat("  FTN offense rolling:", nrow(ftn_off_rolling), "| defense:", nrow(ftn_def_rolling), "\n")
} else {
  cat("  FTN data not available\n")
  ftn_off_rolling <- tibble()
  ftn_def_rolling <- tibble()
}

# ============================================================
# 5. INJURIES (weekly injury impact score)
# ============================================================
cat("Loading injury data...\n")
injury_seasons <- 2012:2025

injuries <- map(injury_seasons, ~ {
  tryCatch(load_injuries(seasons = .x), error = function(e) NULL)
}) |> compact() |> bind_rows()

# Positional value weights for injury impact
pos_value <- c(
  QB = 10, LT = 5, RT = 4, LG = 3, RG = 3, C = 3,
  WR = 4, TE = 3, RB = 3, FB = 1,
  DE = 5, DT = 4, NT = 3, OLB = 4, ILB = 3, MLB = 3, LB = 3,
  CB = 5, SS = 3, FS = 3, S = 3, DB = 3,
  K = 2, P = 1, LS = 0
)

# Status weights: Out = full impact, Doubtful = 0.8, Questionable = 0.3
injury_impact <- injuries |>
  filter(game_type == "REG") |>
  mutate(
    team = normalize_team(team),
    status_weight = case_when(
      report_status == "Out" ~ 1.0,
      report_status == "Doubtful" ~ 0.8,
      report_status == "Questionable" ~ 0.3,
      TRUE ~ 0
    ),
    pos_weight = coalesce(pos_value[position], 1),
    impact = status_weight * pos_weight
  ) |>
  group_by(season, week, team) |>
  summarize(
    injury_impact_score = sum(impact, na.rm = TRUE),
    n_out = sum(report_status == "Out", na.rm = TRUE),
    n_questionable = sum(report_status == "Questionable", na.rm = TRUE),
    qb_out = any(position == "QB" & report_status == "Out", na.rm = TRUE),
    .groups = "drop"
  )

cat("  Injury impact scores:", nrow(injury_impact), "team-weeks\n")

# ============================================================
# 6. PFR ADVANCED STATS (pressure, drops, yards before/after contact)
# ============================================================
cat("Loading PFR advanced stats...\n")
pfr_seasons <- 2018:2025  # PFR available from 2018

pfr_pass <- map(pfr_seasons, ~ {
  tryCatch(load_pfr_advstats(seasons = .x, stat_type = "pass"), error = function(e) NULL)
}) |> compact() |> bind_rows()

pfr_rush <- map(pfr_seasons, ~ {
  tryCatch(load_pfr_advstats(seasons = .x, stat_type = "rush"), error = function(e) NULL)
}) |> compact() |> bind_rows()

pfr_def <- map(pfr_seasons, ~ {
  tryCatch(load_pfr_advstats(seasons = .x, stat_type = "def"), error = function(e) NULL)
}) |> compact() |> bind_rows()

# Aggregate PFR passing to team-week
if (nrow(pfr_pass) > 0) {
  pfr_pass_team <- pfr_pass |>
    filter(game_type == "REG") |>
    mutate(team = normalize_team(team)) |>
    group_by(season, week, team) |>
    summarize(
      pfr_times_pressured = sum(times_pressured, na.rm = TRUE),
      pfr_times_blitzed = sum(times_blitzed, na.rm = TRUE),
      pfr_times_hurried = sum(times_hurried, na.rm = TRUE),
      pfr_times_hit = sum(times_hit, na.rm = TRUE),
      pfr_times_sacked = sum(times_sacked, na.rm = TRUE),
      pfr_drops = sum(passing_drops, na.rm = TRUE),
      pfr_bad_throws = sum(passing_bad_throws, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      pfr_total_dropbacks = pfr_times_pressured + pfr_times_sacked + 20,  # approximate
      pfr_pressure_rate = pfr_times_pressured / pmax(pfr_total_dropbacks, 1),
      pfr_drop_rate = pfr_drops / pmax(pfr_total_dropbacks, 1),
      pfr_bad_throw_rate = pfr_bad_throws / pmax(pfr_total_dropbacks, 1)
    )

  # Rolling PFR features
  pfr_pass_rolling <- pfr_pass_team |>
    group_by(team) |>
    arrange(season, week, .by_group = TRUE) |>
    mutate(
      pfr_roll5_pressure_rate = lag(slide_dbl(pfr_pressure_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
      pfr_roll5_drop_rate = lag(slide_dbl(pfr_drop_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
      pfr_roll5_bad_throw_rate = lag(slide_dbl(pfr_bad_throw_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
    ) |>
    ungroup() |>
    select(season, week, team, starts_with("pfr_roll5_"))

  cat("  PFR passing rolling:", nrow(pfr_pass_rolling), "team-weeks\n")
} else {
  pfr_pass_rolling <- tibble()
}

# Aggregate PFR defense to team-week (pressure generated)
if (nrow(pfr_def) > 0) {
  pfr_def_team <- pfr_def |>
    filter(game_type == "REG") |>
    mutate(team = normalize_team(team)) |>
    group_by(season, week, team) |>
    summarize(
      pfr_def_pressures = sum(def_pressures, na.rm = TRUE),
      pfr_def_targets = sum(def_targets, na.rm = TRUE),
      pfr_def_completions = sum(def_completions_allowed, na.rm = TRUE),
      pfr_def_missed_tackles = sum(def_missed_tackles, na.rm = TRUE),
      pfr_def_sacks = sum(def_sacks, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      pfr_def_total_snaps = pfr_def_targets + pfr_def_sacks + 10,
      pfr_def_pressure_rate = pfr_def_pressures / pmax(pfr_def_total_snaps, 1),
      pfr_def_comp_rate_allowed = pfr_def_completions / pmax(pfr_def_targets, 1),
      pfr_def_missed_tackle_rate = pfr_def_missed_tackles / pmax(pfr_def_targets + 20, 1)
    )

  pfr_def_rolling <- pfr_def_team |>
    group_by(team) |>
    arrange(season, week, .by_group = TRUE) |>
    mutate(
      pfr_roll5_def_pressure = lag(slide_dbl(pfr_def_pressure_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
      pfr_roll5_def_comp_allowed = lag(slide_dbl(pfr_def_comp_rate_allowed, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
    ) |>
    ungroup() |>
    select(season, week, team, starts_with("pfr_roll5_"))

  cat("  PFR defense rolling:", nrow(pfr_def_rolling), "team-weeks\n")
} else {
  pfr_def_rolling <- tibble()
}

# ============================================================
# 7. SITUATIONAL EPA (from PBP - already loaded above)
# ============================================================
cat("Building situational EPA features...\n")

# Situational EPA by team-week (offense)
sit_epa_off <- pbp_plays |>
  mutate(posteam = normalize_team(posteam)) |>
  group_by(season, week, posteam) |>
  summarize(
    sit_pass_epa = mean(epa[pass == 1], na.rm = TRUE),
    sit_rush_epa = mean(epa[rush == 1], na.rm = TRUE),
    sit_3rd_down_epa = mean(epa[down == 3], na.rm = TRUE),
    sit_redzone_epa = mean(epa[yardline_100 <= 20], na.rm = TRUE),
    sit_early_down_epa = mean(epa[down %in% c(1, 2)], na.rm = TRUE),
    sit_explosive_rate = mean(epa > 1.5, na.rm = TRUE),  # high-EPA plays
    sit_neg_play_rate = mean(epa < -1, na.rm = TRUE),     # disasters
    sit_n_plays = n(),
    .groups = "drop"
  )

# Situational EPA by team-week (defense)
sit_epa_def <- pbp_plays |>
  mutate(defteam = normalize_team(defteam)) |>
  group_by(season, week, defteam) |>
  summarize(
    sit_def_pass_epa = mean(epa[pass == 1], na.rm = TRUE),
    sit_def_rush_epa = mean(epa[rush == 1], na.rm = TRUE),
    sit_def_3rd_epa = mean(epa[down == 3], na.rm = TRUE),
    sit_def_redzone_epa = mean(epa[yardline_100 <= 20], na.rm = TRUE),
    sit_def_explosive_rate = mean(epa > 1.5, na.rm = TRUE),
    .groups = "drop"
  )

# Rolling situational EPA (lagged)
sit_off_rolling <- sit_epa_off |>
  group_by(posteam) |>
  arrange(season, week, .by_group = TRUE) |>
  mutate(
    sit_roll5_pass_epa = lag(slide_dbl(sit_pass_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_rush_epa = lag(slide_dbl(sit_rush_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_3rd_epa = lag(slide_dbl(sit_3rd_down_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_rz_epa = lag(slide_dbl(sit_redzone_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_explosive = lag(slide_dbl(sit_explosive_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_neg_play = lag(slide_dbl(sit_neg_play_rate, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
  ) |>
  ungroup() |>
  select(season, week, posteam, starts_with("sit_roll5_"))

sit_def_rolling <- sit_epa_def |>
  group_by(defteam) |>
  arrange(season, week, .by_group = TRUE) |>
  mutate(
    sit_roll5_def_pass_epa = lag(slide_dbl(sit_def_pass_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_def_rush_epa = lag(slide_dbl(sit_def_rush_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_def_3rd_epa = lag(slide_dbl(sit_def_3rd_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE)),
    sit_roll5_def_rz_epa = lag(slide_dbl(sit_def_redzone_epa, ~ mean(.x, na.rm = TRUE), .before = 4, .complete = FALSE))
  ) |>
  ungroup() |>
  select(season, week, defteam, starts_with("sit_roll5_"))

cat("  Situational EPA - offense:", nrow(sit_off_rolling), "| defense:", nrow(sit_def_rolling), "\n")

# ============================================================
# 8. MERGE EVERYTHING INTO FAV/DOG PERSPECTIVE
# ============================================================
cat("\nMerging all features into fav/dog perspective...\n")

# Start with the existing model_data and join advanced features
# We need to map game_id_norm -> fav_team, dog_team from team_game_full
fav_rows_map <- team_game_full |>
  filter(cover != "push", team_norm == team_favorite_id) |>
  select(game_id_norm, team_norm) |>
  rename(team_fav = team_norm) |>
  mutate(team_fav = as.character(team_fav)) |>
  distinct(game_id_norm, .keep_all = TRUE)

dog_rows_map <- team_game_full |>
  filter(cover != "push", team_norm != team_favorite_id) |>
  select(game_id_norm, team_norm) |>
  rename(team_dog = team_norm) |>
  mutate(team_dog = as.character(team_dog)) |>
  distinct(game_id_norm, .keep_all = TRUE)

fav_dog_map <- fav_rows_map |> inner_join(dog_rows_map, by = "game_id_norm")

# Get season/week for each game
game_info <- model_data |> select(game_id_norm, season, week) |>
  mutate(season = as.numeric(season), week = as.numeric(week))

# Ensure all join sources have consistent types
ensure_types <- function(df) {
  if ("season" %in% names(df)) df <- df |> mutate(season = as.numeric(season))
  if ("week" %in% names(df)) df <- df |> mutate(week = as.numeric(week))
  df
}

qb_rolling <- ensure_types(qb_rolling)
ngs_team <- ensure_types(ngs_team)
sit_off_rolling <- ensure_types(sit_off_rolling)
sit_def_rolling <- ensure_types(sit_def_rolling)
injury_impact <- ensure_types(injury_impact)
if (nrow(pfr_pass_rolling) > 0) pfr_pass_rolling <- ensure_types(pfr_pass_rolling)
if (nrow(pfr_def_rolling) > 0) pfr_def_rolling <- ensure_types(pfr_def_rolling)
if (nrow(ftn_off_rolling) > 0) ftn_off_rolling <- ensure_types(ftn_off_rolling)
if (nrow(ftn_def_rolling) > 0) ftn_def_rolling <- ensure_types(ftn_def_rolling)
schedule_features <- ensure_types(schedule_features)

advanced_features <- game_info |>
  left_join(fav_dog_map, by = "game_id_norm") |>

  # Schedule features
  left_join(schedule_features |> select(game_id_norm, div_game, is_primetime, is_dome, is_turf),
            by = "game_id_norm") |>

  # QB features (fav)
  left_join(qb_rolling |> rename(fav_qb_roll5_epa = qb_roll5_epa,
                                  fav_qb_roll5_cpoe = qb_roll5_cpoe,
                                  fav_qb_roll5_sack_rate = qb_roll5_sack_rate,
                                  fav_qb_roll5_air_yards = qb_roll5_air_yards),
            by = c("season", "week", "team_fav" = "team")) |>
  # QB features (dog)
  left_join(qb_rolling |> rename(dog_qb_roll5_epa = qb_roll5_epa,
                                  dog_qb_roll5_cpoe = qb_roll5_cpoe,
                                  dog_qb_roll5_sack_rate = qb_roll5_sack_rate,
                                  dog_qb_roll5_air_yards = qb_roll5_air_yards),
            by = c("season", "week", "team_dog" = "team")) |>

  # NGS features (fav)
  left_join(ngs_team |> rename_with(~ paste0("fav_", .), starts_with("ngs_")),
            by = c("season", "week", "team_fav" = "team_abbr")) |>
  # NGS features (dog)
  left_join(ngs_team |> rename_with(~ paste0("dog_", .), starts_with("ngs_")),
            by = c("season", "week", "team_dog" = "team_abbr")) |>

  # Situational EPA (fav offense)
  left_join(sit_off_rolling |> rename_with(~ paste0("fav_off_", .), starts_with("sit_roll5_")),
            by = c("season", "week", "team_fav" = "posteam")) |>
  # Situational EPA (dog offense)
  left_join(sit_off_rolling |> rename_with(~ paste0("dog_off_", .), starts_with("sit_roll5_")),
            by = c("season", "week", "team_dog" = "posteam")) |>

  # Situational EPA (fav defense)
  left_join(sit_def_rolling |> rename_with(~ paste0("fav_def_", .), starts_with("sit_roll5_")),
            by = c("season", "week", "team_fav" = "defteam")) |>
  # Situational EPA (dog defense)
  left_join(sit_def_rolling |> rename_with(~ paste0("dog_def_", .), starts_with("sit_roll5_")),
            by = c("season", "week", "team_dog" = "defteam")) |>

  # Injuries (fav and dog)
  left_join(injury_impact |> rename(fav_injury_impact = injury_impact_score,
                                     fav_n_out = n_out,
                                     fav_qb_out = qb_out),
            by = c("season", "week", "team_fav" = "team")) |>
  left_join(injury_impact |> rename(dog_injury_impact = injury_impact_score,
                                     dog_n_out = n_out,
                                     dog_qb_out = qb_out),
            by = c("season", "week", "team_dog" = "team")) |>

  # PFR pass rolling (fav and dog)
  left_join(pfr_pass_rolling |> rename_with(~ paste0("fav_", .), starts_with("pfr_roll5_")),
            by = c("season", "week", "team_fav" = "team")) |>
  left_join(pfr_pass_rolling |> rename_with(~ paste0("dog_", .), starts_with("pfr_roll5_")),
            by = c("season", "week", "team_dog" = "team")) |>

  # PFR defense rolling (fav and dog)
  left_join(pfr_def_rolling |> rename_with(~ paste0("fav_", .), starts_with("pfr_roll5_")),
            by = c("season", "week", "team_fav" = "team")) |>
  left_join(pfr_def_rolling |> rename_with(~ paste0("dog_", .), starts_with("pfr_roll5_")),
            by = c("season", "week", "team_dog" = "team"))

# FTN features if available
if (nrow(ftn_off_rolling) > 0) {
  advanced_features <- advanced_features |>
    left_join(ftn_off_rolling |> rename_with(~ paste0("fav_", .), starts_with("ftn_roll5_")),
              by = c("season", "week", "team_fav" = "posteam")) |>
    left_join(ftn_off_rolling |> rename_with(~ paste0("dog_", .), starts_with("ftn_roll5_")),
              by = c("season", "week", "team_dog" = "posteam")) |>
    left_join(ftn_def_rolling |> rename_with(~ paste0("fav_", .), starts_with("ftn_roll5_")),
              by = c("season", "week", "team_fav" = "defteam")) |>
    left_join(ftn_def_rolling |> rename_with(~ paste0("dog_", .), starts_with("ftn_roll5_")),
              by = c("season", "week", "team_dog" = "defteam"))
}

# ============================================================
# 9. BUILD DIFFERENTIAL FEATURES
# ============================================================
cat("Building differential features...\n")

advanced_diffs <- advanced_features |>
  mutate(
    # QB mismatch
    qb_epa_diff = fav_qb_roll5_epa - dog_qb_roll5_epa,
    qb_cpoe_diff = fav_qb_roll5_cpoe - dog_qb_roll5_cpoe,
    qb_sack_rate_diff = fav_qb_roll5_sack_rate - dog_qb_roll5_sack_rate,

    # Situational EPA differentials
    pass_epa_mismatch = fav_off_sit_roll5_pass_epa - dog_def_sit_roll5_def_pass_epa,  # fav pass O vs dog pass D
    rush_epa_mismatch = fav_off_sit_roll5_rush_epa - dog_def_sit_roll5_def_rush_epa,  # fav rush O vs dog rush D
    third_down_epa_diff = fav_off_sit_roll5_3rd_epa - dog_off_sit_roll5_3rd_epa,
    redzone_epa_diff = fav_off_sit_roll5_rz_epa - dog_off_sit_roll5_rz_epa,
    explosive_diff = fav_off_sit_roll5_explosive - dog_off_sit_roll5_explosive,
    neg_play_diff = fav_off_sit_roll5_neg_play - dog_off_sit_roll5_neg_play,

    # Injury differential
    injury_diff = coalesce(dog_injury_impact, 0) - coalesce(fav_injury_impact, 0),
    fav_qb_out = coalesce(as.integer(fav_qb_out), 0L),
    dog_qb_out = coalesce(as.integer(dog_qb_out), 0L),

    # Context
    div_game = coalesce(as.integer(div_game), 0L),
    is_primetime = coalesce(as.integer(is_primetime), 0L),
    is_dome = coalesce(as.integer(is_dome), 0L)
  )

# ============================================================
# 10. MERGE WITH ORIGINAL MODEL DATA AND SAVE
# ============================================================
cat("Merging with original model_data...\n")

# Select the new features we want to add
new_features <- advanced_diffs |>
  select(
    game_id_norm,
    # Context
    div_game, is_primetime, is_dome,
    # QB quality
    qb_epa_diff, qb_cpoe_diff, qb_sack_rate_diff,
    fav_qb_roll5_epa, dog_qb_roll5_epa,
    # Situational EPA
    pass_epa_mismatch, rush_epa_mismatch,
    third_down_epa_diff, redzone_epa_diff, explosive_diff, neg_play_diff,
    fav_off_sit_roll5_pass_epa, fav_off_sit_roll5_rush_epa,
    dog_off_sit_roll5_pass_epa, dog_off_sit_roll5_rush_epa,
    # Injury
    injury_diff, fav_qb_out, dog_qb_out,
    # NGS (where available)
    starts_with("fav_ngs_"), starts_with("dog_ngs_"),
    # PFR pressure
    starts_with("fav_pfr_"), starts_with("dog_pfr_"),
    # FTN
    starts_with("fav_ftn_"), starts_with("dog_ftn_")
  )

model_data_v2 <- model_data |>
  left_join(new_features, by = "game_id_norm")

cat("\n=== FINAL DATASET ===\n")
cat("Rows:", nrow(model_data_v2), "\n")
cat("Columns:", ncol(model_data_v2), "\n")
cat("New columns added:", ncol(model_data_v2) - ncol(model_data), "\n")

# Check missingness of new features
new_col_names <- setdiff(names(model_data_v2), names(model_data))
miss_report <- model_data_v2 |>
  select(all_of(new_col_names)) |>
  summarize(across(everything(), ~ round(mean(is.na(.)) * 100, 1))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "pct_missing") |>
  arrange(pct_missing)

cat("\nMissingness of new features:\n")
print(miss_report, n = 50)

# Check correlations with outcome
outcome_num <- as.integer(model_data_v2$fav_cover == "cover")
cors <- model_data_v2 |>
  select(all_of(new_col_names)) |>
  select(where(is.numeric)) |>
  map_dbl(~ cor(.x, outcome_num, use = "complete.obs")) |>
  sort(decreasing = TRUE)

cat("\nCorrelations of new features with cover outcome:\n")
print(round(cors, 4))

# Save
write_rds(model_data_v2, here("objects", "modeling_data_v2.rds"))
cat("\nSaved: objects/modeling_data_v2.rds\n")
cat("Done!\n")
