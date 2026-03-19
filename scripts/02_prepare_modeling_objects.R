# Prep ing train/test split, CV folds, and recipes

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "00_project_setup.R"))

model_data <- read_rds(here("objects", "modeling_data.rds"))

# Temporal split: 2012-2020 train, 2021+ test
train_data <- model_data |> filter(season <= 2020)
test_data  <- model_data |> filter(season >= 2021)

# 10-fold CV with 3 repeats, stratified by outcome
set.seed(301)
cv_folds <- vfold_cv(train_data, v = 10, repeats = 3, strata = fav_cover)

# Recipe 1: Kitchen Sink (all predictors, minimal preprocessing)
recipe_basic_linear <- recipe(fav_cover ~ ., data = train_data) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_basic_tree <- recipe(fav_cover ~ ., data = train_data) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())

# Recipe 2: Feature Engineered (domain-specific interactions)
recipe_engineered_linear <- recipe(fav_cover ~ ., data = train_data) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_mutate(
    abs_spread = abs(spread_fav),
    market_pressure = abs(spread_fav) * over_under_line / 100,
    wind_cold = if_else(weather_temperature < 45, weather_wind_mph, 0),
    rest_edge = pmax(rest_diff, 0),
    favorite_home_spread = if_else(fav_at_home == "yes", spread_fav, 0),
    strength_signal = elo_diff * pts_play_diff,
    market_signal_gap = elo_diff - abs_spread,
    pass_rush_balance = pass_eff_diff - rush_eff_diff,
    epa_signal = off_epa_diff - def_epa_diff
  ) |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_interact(
    terms = ~ abs_spread:elo_diff + favorite_home_spread:elo_diff + weather_wind_mph:pass_eff_diff
  ) |>
  step_corr(all_numeric_predictors(), threshold = 0.85) |>
  step_normalize(all_numeric_predictors())

# Engineered tree

recipe_engineered_tree <- recipe(fav_cover ~ ., data = train_data) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_mutate(
    abs_spread = abs(spread_fav),
    market_pressure = abs(spread_fav) * over_under_line / 100,
    wind_cold = if_else(weather_temperature < 45, weather_wind_mph, 0),
    rest_edge = pmax(rest_diff, 0),
    favorite_home_spread = if_else(fav_at_home == "yes", spread_fav, 0),
    strength_signal = elo_diff * pts_play_diff,
    market_signal_gap = elo_diff - abs_spread,
    pass_rush_balance = pass_eff_diff - rush_eff_diff,
    epa_signal = off_epa_diff - def_epa_diff
  ) |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.85)

# Verify recipes process correctly
recipe_basic_linear |> prep() |> bake(new_data = NULL) |> glimpse()
recipe_basic_tree |> prep() |> bake(new_data = NULL) |> glimpse()
recipe_engineered_linear |> prep() |> bake(new_data = NULL) |> glimpse()
recipe_engineered_tree |> prep() |> bake(new_data = NULL) |> glimpse()

# Recipe 3: Curated (top 10 predictors from initial Lasso analysis)
curated_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  "fav_roll5_ypp", "ypp_diff", "turnover_diff", "dog_roll5_pts",
  "weather_temperature", "off_epa_diff", "dog_short_week",
  "elo_diff", "def_epa_diff", "to_momentum_diff", "success_rate_diff"
)

recipe_curated_linear <- recipe(fav_cover ~ ., data = train_data |> select(any_of(curated_vars))) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_curated_tree <- recipe(fav_cover ~ ., data = train_data |> select(any_of(curated_vars))) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())

recipe_curated_linear |> prep() |> bake(new_data = NULL) |> glimpse()
recipe_curated_tree |> prep() |> bake(new_data = NULL) |> glimpse()

# Save prepared objects
prepared_objects <- list(
  train_data = train_data,
  test_data = test_data,
  cv_folds = cv_folds,
  recipes = list(
    basic_linear = recipe_basic_linear,
    basic_tree = recipe_basic_tree,
    engineered_linear = recipe_engineered_linear,
    engineered_tree = recipe_engineered_tree,
    curated_linear = recipe_curated_linear,
    curated_tree = recipe_curated_tree
  )
)

split_summary <- tibble(
  measure = c("Training rows", "Test rows", "Cross-validation folds", "Cross-validation repeats"),
  value = c(nrow(train_data), nrow(test_data), 10, 3)
)

write_rds(prepared_objects, here("objects", "prepared_modeling_objects.rds"))
write_csv(split_summary, here("results", "split_summary.csv"))
