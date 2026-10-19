# Shared setup for the NFL spread prediction project

library(tidyverse)
library(tidymodels)
library(here)
library(janitor)
library(lubridate)
library(nflreadr)
library(slider)
library(xgboost)
library(future)
library(broom)
library(knitr)
library(conflicted)

conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
tidymodels_prefer()

# Parallel processing
available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

# Create output directories
dir.create(here("objects"), showWarnings = FALSE)
dir.create(here("results"), showWarnings = FALSE)
dir.create(here("results", "workflows"), showWarnings = FALSE)
dir.create(here("figures"), showWarnings = FALSE)

# Writing in all the team name lookup for franchise relocations 
team_lookup <- tribble(
  ~team_name,                  ~team,
  "Arizona Cardinals",          "ARI",
  "Atlanta Falcons",            "ATL",
  "Baltimore Ravens",           "BAL",
  "Buffalo Bills",              "BUF",
  "Carolina Panthers",          "CAR",
  "Chicago Bears",              "CHI",
  "Cincinnati Bengals",         "CIN",
  "Cleveland Browns",           "CLE",
  "Dallas Cowboys",             "DAL",
  "Denver Broncos",             "DEN",
  "Detroit Lions",              "DET",
  "Green Bay Packers",          "GB",
  "Houston Texans",             "HOU",
  "Indianapolis Colts",         "IND",
  "Jacksonville Jaguars",       "JAX",
  "Kansas City Chiefs",         "KC",
  "Las Vegas Raiders",          "LVR",
  "Oakland Raiders",            "LVR",
  "Los Angeles Chargers",       "LAC",
  "San Diego Chargers",         "LAC",
  "Los Angeles Rams",           "LAR",
  "St. Louis Rams",             "LAR",
  "Miami Dolphins",             "MIA",
  "Minnesota Vikings",          "MIN",
  "New England Patriots",       "NE",
  "New Orleans Saints",         "NO",
  "New York Giants",            "NYG",
  "New York Jets",              "NYJ",
  "Philadelphia Eagles",        "PHI",
  "Pittsburgh Steelers",        "PIT",
  "San Francisco 49ers",        "SF",
  "Seattle Seahawks",           "SEA",
  "Tampa Bay Buccaneers",       "TB",
  "Tennessee Titans",           "TEN",
  "Washington Redskins",        "WAS",
  "Washington Football Team",   "WAS",
  "Washington Commanders",      "WAS"
)

# I re-learned how to R helper functions

normalize_game_id <- function(x) {
  x |>
    str_replace_all("_SD_",  "_LAC_") |>
    str_replace_all("_SD$",  "_LAC")  |>
    str_replace_all("_OAK_", "_LVR_") |>
    str_replace_all("_OAK$", "_LVR")  |>
    str_replace_all("_LV_",  "_LVR_") |>
    str_replace_all("_LV$",  "_LVR")  |>
    str_replace_all("_STL_", "_LAR_") |>
    str_replace_all("_STL$", "_LAR")  |>
    str_replace_all("_LA_",  "_LAR_") |>
    str_replace_all("_LA$",  "_LAR")
}

normalize_team <- function(x) {
  case_when(
    x == "LV"  ~ "LVR",
    x == "LA"  ~ "LAR",
    x == "OAK" ~ "LVR",
    x == "SD"  ~ "LAC",
    x == "STL" ~ "LAR",
    x == "JAC" ~ "JAX",
    TRUE ~ x
  )
}

lagged_roll <- function(x, window) {
  lag(
    slide_dbl(
      x,
      ~ mean(.x, na.rm = TRUE),
      .before = window - 1,
      .after = 0,
      .complete = FALSE
    )
  )
}

# Applying my ELO Learnings

compute_elo <- function(games_df, k_value = 20, regress = 0.75) {
  teams <- unique(c(games_df$home_team, games_df$away_team))
  elo <- setNames(rep(1500, length(teams)), teams)

  n_games <- nrow(games_df)
  result <- tibble(
    game_id_norm = rep(NA_character_, 2 * n_games),
    team = rep(NA_character_, 2 * n_games),
    pre_elo = rep(NA_real_, 2 * n_games)
  )

  previous_season <- games_df$season[1] - 1L

  for (i in seq_len(n_games)) {
    current_season <- games_df$season[i]

    if (current_season != previous_season) {
      elo <- elo * regress + 1500 * (1 - regress)
      previous_season <- current_season
    }

    home_team <- games_df$home_team[i]
    away_team <- games_df$away_team[i]
    row_index <- (i - 1L) * 2L

    result$game_id_norm[row_index + 1L] <- games_df$game_id_norm[i]
    result$team[row_index + 1L] <- home_team
    result$pre_elo[row_index + 1L] <- elo[home_team]

    result$game_id_norm[row_index + 2L] <- games_df$game_id_norm[i]
    result$team[row_index + 2L] <- away_team
    result$pre_elo[row_index + 2L] <- elo[away_team]

    score_home <- games_df$score_home[i]
    score_away <- games_df$score_away[i]

    if (is.na(score_home) || is.na(score_away)) {
      next
    }

    expected_home <- 1 / (1 + 10^((elo[away_team] - elo[home_team]) / 400))
    actual_home <- if (score_home > score_away) 1 else if (score_home < score_away) 0 else 0.5

    elo[home_team] <- elo[home_team] + k_value * (actual_home - expected_home)
    elo[away_team] <- elo[away_team] + k_value * ((1 - actual_home) - (1 - expected_home))
  }

  filter(result, !is.na(game_id_norm))
}

safe_ratio <- function(numerator, denominator, fallback = 0) {
  ifelse(is.na(denominator) | denominator == 0, fallback, numerator / denominator)
}

format_parameter_string <- function(best_params) {
  if (nrow(best_params) == 0) return("none")

  param_cols <- setdiff(names(best_params), c(".config", ".iter"))
  if (length(param_cols) == 0) return("none")

  values <- map2_chr(
    param_cols,
    best_params[1, param_cols, drop = TRUE],
    ~ paste0(.x, "=", signif(as.numeric(.y), 4))
  )

  paste(values, collapse = "; ")
}

## Workflow summary


extract_workflow_summary <- function(workflow_row, workflow_result) {
  if (workflow_row$tuned) {
    best_params <- select_best(workflow_result, metric = "roc_auc")
    metric_table <- collect_metrics(workflow_result) |>
      semi_join(
        best_params,
        by = intersect(names(best_params), names(collect_metrics(workflow_result)))
      ) |>
      filter(.metric %in% c("roc_auc", "accuracy")) |>
      select(.metric, mean, std_err)
    hyperparameters <- format_parameter_string(best_params)
  } else {
    best_params <- tibble()
    metric_table <- collect_metrics(workflow_result) |>
      filter(.metric %in% c("roc_auc", "accuracy")) |>
      select(.metric, mean, std_err)
    hyperparameters <- "none"
  }

  metric_wide <- metric_table |>
    mutate(
      metric_name = case_when(
        .metric == "roc_auc" ~ "roc_auc",
        .metric == "accuracy" ~ "accuracy",
        TRUE ~ .metric
      )
    ) |>
    select(metric_name, mean, std_err) |>
    pivot_wider(
      names_from = metric_name,
      values_from = c(mean, std_err),
      names_glue = "{metric_name}_{.value}"
    )

  bind_cols(
    select(workflow_row, workflow_id, workflow_label, model_type, recipe_family, tuned),
    metric_wide
  ) |>
    mutate(
      hyperparameters = hyperparameters,
      roc_auc = roc_auc_mean,
      roc_auc_std_err = roc_auc_std_err,
      accuracy = accuracy_mean
    ) |>
    select(
      workflow_id, workflow_label, model_type, recipe_family, tuned,
      roc_auc, roc_auc_std_err, accuracy, hyperparameters
    )
}

## Workflow fit

finalize_workflow_fit <- function(workflow_object, workflow_result, train_data) {
  has_tunable_parameters <- !is.null(attr(workflow_result, "parameters")) &&
    nrow(attr(workflow_result, "parameters")) > 0

  if (has_tunable_parameters) {
    best_params <- select_best(workflow_result, metric = "roc_auc")
    final_workflow <- finalize_workflow(workflow_object, best_params)
  } else {
    best_params <- tibble()
    final_workflow <- workflow_object
  }

  list(
    workflow = final_workflow,
    fit = fit(final_workflow, data = train_data),
    best_params = best_params
  )
}

combine_model_bundles <- function(bundles, metrics, control_grid, control_resamples) {
  list(
    metrics = metrics,
    control_grid = control_grid,
    control_resamples = control_resamples,
    workflow_info = bind_rows(map(bundles, "workflow_info")),
    specs = flatten(map(bundles, "specs")),
    workflows = flatten(map(bundles, "workflows")),
    grids = flatten(map(bundles, "grids"))
  )
}

fit_workflow_set <- function(workflow_ids, prepared_objects, workflow_objects, seed = 301) {
  cv_folds <- prepared_objects$cv_folds
  workflow_info <- workflow_objects$workflow_info |>
    filter(workflow_id %in% workflow_ids)
  workflows <- workflow_objects$workflows
  grids <- workflow_objects$grids

  set.seed(seed)

  workflow_results <- map(
    seq_len(nrow(workflow_info)),
    function(i) {
      workflow_row <- workflow_info[i, ]
      wf_id <- workflow_row$workflow_id
      wf <- workflows[[wf_id]]

      if (workflow_row$tuned) {
        tune_grid(
          wf,
          resamples = cv_folds,
          grid = grids[[wf_id]],
          metrics = workflow_objects$metrics,
          control = workflow_objects$control_grid
        )
      } else {
        fit_resamples(
          wf,
          resamples = cv_folds,
          metrics = workflow_objects$metrics,
          control = workflow_objects$control_resamples
        )
      }
    }
  )

  names(workflow_results) <- workflow_info$workflow_id
  workflow_results
}

# Add empty team-week rows for upcoming games (objects/upcoming_games.rds,
# written by 02_prepare_recent_data.R) so lagged rolling features land on
# games that haven't been played yet. Both nflverse and normalized team
# abbreviations are added since source tables differ.
with_upcoming <- function(df, team_col) {
  path <- here("objects", "upcoming_games.rds")
  if (!file.exists(path)) return(df)
  up <- read_rds(path)
  teams <- c(up$home_team, up$away_team)
  raw_alias <- c(LAR = "LA", LVR = "LV")
  rows <- tibble(season = rep(c(up$season, up$season), 2),
                 week = rep(c(up$week, up$week), 2),
                 team = c(teams, coalesce(unname(raw_alias[teams]), teams))) |>
    distinct()
  names(rows)[3] <- team_col
  rows <- rows |> mutate(season = as(season, class(df$season)[1]), week = as(week, class(df$week)[1]))
  bind_rows(df, anti_join(rows, df, by = c("season", "week", team_col)))
}
