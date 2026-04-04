# Baseline and logistic regression model specs

null_spec <- null_model() |>
  set_engine("parsnip") |>
  set_mode("classification")

log_spec <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

# Define workflow metadata (one row per workflow)
workflow_info <- bind_rows(
  tibble(
    workflow_id = "null_basic",
    workflow_label = "Baseline",
    model_type = "Null baseline",
    recipe_family = "Basic",
    recipe_key = "basic_linear",
    model_key = "null",
    tuned = FALSE
  ),
  tibble(
    workflow_id = "logistic_basic",
    workflow_label = "Logistic Regression - Basic",
    model_type = "Logistic regression",
    recipe_family = "Basic",
    recipe_key = "basic_linear",
    model_key = "log",
    tuned = FALSE
  ),
  tibble(
    workflow_id = "logistic_engineered",
    workflow_label = "Logistic Regression - Engineered",
    model_type = "Logistic regression",
    recipe_family = "Engineered",
    recipe_key = "engineered_linear",
    model_key = "log",
    tuned = FALSE
  ),
  tibble(
    workflow_id = "logistic_curated",
    workflow_label = "Logistic Regression - Curated",
    model_type = "Logistic regression",
    recipe_family = "Curated",
    recipe_key = "curated_linear",
    model_key = "log",
    tuned = FALSE
  )
)

# Assemble bundle
model_bundle_baseline_logistic <- list(
  workflow_info = workflow_info,
  specs = list(
    null = null_spec,
    log = log_spec
  ),
  workflows = list(
    null_basic = workflow() |>
      add_model(null_spec) |>
      add_recipe(recipes$basic_linear),
    logistic_basic = workflow() |>
      add_model(log_spec) |>
      add_recipe(recipes$basic_linear),
    logistic_engineered = workflow() |>
      add_model(log_spec) |>
      add_recipe(recipes$engineered_linear),
    logistic_curated = workflow() |>
      add_model(log_spec) |>
      add_recipe(recipes$curated_linear)
  ),
  grids = list()
)
