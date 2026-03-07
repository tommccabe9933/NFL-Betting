library(tidyverse)
library(janitor)
library(readr)
library(tidymodels)
library(stringr)
library(here)

tidymodels_prefer()

dir.create(here("data"), showWarnings = FALSE)

team_offense <- read_csv(here("data","weekly_team_stats_offense.csv"), show_col_types = FALSE) |>
  clean_names()

team_defense <- read_csv(here("data","weekly_team_stats_defense.csv"), show_col_types = FALSE) |>
  clean_names()

games_all <- read_csv(here("data","nfl_all_seasons.csv"), show_col_types = FALSE) |>
  clean_names()

team_lookup <- tribble(
  ~team_name, ~team,
  "Arizona Cardinals","ARI",
  "Atlanta Falcons","ATL",
  "Baltimore Ravens","BAL",
  "Buffalo Bills","BUF",
  "Carolina Panthers","CAR",
  "Chicago Bears","CHI",
  "Cincinnati Bengals","CIN",
  "Cleveland Browns","CLE",
  "Dallas Cowboys","DAL",
  "Denver Broncos","DEN",
  "Detroit Lions","DET",
  "Green Bay Packers","GB",
  "Houston Texans","HOU",
  "Indianapolis Colts","IND",
  "Jacksonville Jaguars","JAX",
  "Kansas City Chiefs","KC",
  "Las Vegas Raiders","LVR",
  "Oakland Raiders","LVR",
  "Los Angeles Chargers","LAC",
  "San Diego Chargers","LAC",
  "Los Angeles Rams","LAR",
  "St. Louis Rams","LAR",
  "Miami Dolphins","MIA",
  "Minnesota Vikings","MIN",
  "New England Patriots","NE",
  "New Orleans Saints","NO",
  "New York Giants","NYG",
  "New York Jets","NYJ",
  "Philadelphia Eagles","PHI",
  "Pittsburgh Steelers","PIT",
  "San Francisco 49ers","SF",
  "Seattle Seahawks","SEA",
  "Tampa Bay Buccaneers","TB",
  "Tennessee Titans","TEN",
  "Washington Redskins","WAS",
  "Washington Football Team","WAS",
  "Washington Commanders","WAS"
)

normalize_game_id <- function(x){
  x |>
    str_replace_all("_SD_","_LAC_") |>
    str_replace_all("_OAK_","_LVR_") |>
    str_replace_all("_STL_","_LAR_") |>
    str_replace_all("_LA_","_LAR_")
}

normalize_team <- function(x){
  case_when(
    x=="LV"~"LVR",
    x=="LA"~"LAR",
    TRUE~x
  )
}

games_reg <- games_all |>
  filter(schedule_season>=2012, schedule_playoff==FALSE) |>
  mutate(schedule_week=as.integer(schedule_week)) |>
  left_join(team_lookup, by=c("team_home"="team_name")) |>
  rename(home_team=team) |>
  left_join(team_lookup, by=c("team_away"="team_name")) |>
  rename(away_team=team) |>
  mutate(
    team_favorite_id=normalize_team(team_favorite_id),
    game_id_norm=str_c(
      schedule_season,"_",
      str_pad(schedule_week,2,pad="0"),"_",
      away_team,"_",home_team
    )
  )

games_team <- games_reg |>
  transmute(
    game_id_norm,
    season=schedule_season,
    week=schedule_week,
    season_type="REG",
    team_norm=home_team,
    opponent=away_team,
    home_away="home",
    team_favorite_id,
    team_score=score_home,
    opponent_score=score_away,
    spread_favorite,
    over_under_line,
    stadium,
    stadium_neutral,
    weather_temperature,
    weather_wind_mph,
    weather_humidity
  ) |>
  bind_rows(
    games_reg |>
      transmute(
        game_id_norm,
        season=schedule_season,
        week=schedule_week,
        season_type="REG",
        team_norm=away_team,
        opponent=home_team,
        home_away="away",
        team_favorite_id,
        team_score=score_away,
        opponent_score=score_home,
        spread_favorite,
        over_under_line,
        stadium,
        stadium_neutral,
        weather_temperature,
        weather_wind_mph,
        weather_humidity
      )
  )

team_offense_clean <- team_offense |>
  mutate(
    game_id_norm=normalize_game_id(game_id),
    team_norm=normalize_team(team)
  )

team_defense_clean <- team_defense |>
  mutate(
    game_id_norm=normalize_game_id(game_id),
    team_norm=normalize_team(team)
  )

model_data <- games_team |>
  left_join(
    team_offense_clean,
    by=c("game_id_norm","season","week","season_type","team_norm")
  ) |>
  left_join(
    team_defense_clean,
    by=c("game_id_norm","season","week","season_type","team_norm"),
    suffix=c("_off","_def")
  )

model_data <- model_data |>
  mutate(
    spread_team=if_else(team_norm==team_favorite_id,spread_favorite,-spread_favorite),
    margin=team_score-opponent_score,
    margin_vs_spread=margin+spread_team,
    cover=case_when(
      margin_vs_spread>0~"cover",
      margin_vs_spread<0~"no_cover",
      TRUE~"push"
    )
  ) |>
  filter(cover!="push") |>
  mutate(
    cover=factor(cover),
    home_away=factor(home_away),
    stadium_neutral=factor(stadium_neutral)
  )

tom_cleaned_model_nfl <- model_data |>
  select(
    cover,
    home_away,
    stadium_neutral,
    spread_team,
    over_under_line,
    weather_temperature,
    weather_wind_mph,
    weather_humidity,
    win_pct_off,
    average_total_off_yards,
    average_pass_attempts,
    average_complete_pass,
    average_passing_yards,
    average_rush_attempts,
    average_rushing_yards,
    average_first_down_pass,
    average_first_down_rush,
    average_third_down_converted,
    average_third_down_failed,
    average_rush_touchdown,
    average_pass_touchdown,
    average_interception_off,
    average_fumble_lost_off,
    average_sack,
    average_qb_hit,
    average_interception_def,
    average_fumble_forced_def,
    average_solo_tackle,
    average_assist_tackle
  ) |>
  filter(
    !is.na(cover),
    !is.na(spread_team),
    !is.na(over_under_line),
    !is.na(average_total_off_yards)
  )

data_split <- initial_split(
  tom_cleaned_model_nfl,
  prop = 0.80,
  strata = cover
)

train_data <- training(data_split)
test_data <- testing(data_split)

cv_folds <- vfold_cv(
  train_data,
  v = 10,
  strata = cover
)