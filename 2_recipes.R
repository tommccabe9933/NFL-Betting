library(tidymodels)

recipe_linear <- recipe(cover ~ ., data = train_data) |>
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_tree <- recipe(cover ~ ., data = train_data) |>
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors())