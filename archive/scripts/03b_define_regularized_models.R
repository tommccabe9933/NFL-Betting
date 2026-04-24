# Regularized model specs: Lasso, Ridge, Elastic Net

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("classification")

ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("classification")

enet_spec <- logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("classification")

# Define workflow metadata (one row per workflow)
workflow_info <- bind_rows(
  tibble(
    workflow_id = "lasso_basic",
    workflow_label = "Lasso - Basic",
    model_type = "Lasso",
    recipe_family = "Basic",
    recipe_key = "basic_linear",
    model_key = "lasso",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "lasso_engineered",
    workflow_label = "Lasso - Engineered",
    model_type = "Lasso",
    recipe_family = "Engineered",
    recipe_key = "engineered_linear",
    model_key = "lasso",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "ridge_basic",
    workflow_label = "Ridge - Basic",
    model_type = "Ridge",
    recipe_family = "Basic",
    recipe_key = "basic_linear",
    model_key = "ridge",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "ridge_engineered",
    workflow_label = "Ridge - Engineered",
    model_type = "Ridge",
    recipe_family = "Engineered",
    recipe_key = "engineered_linear",
    model_key = "ridge",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "enet_basic",
    workflow_label = "Elastic Net - Basic",
    model_type = "Elastic net",
    recipe_family = "Basic",
    recipe_key = "basic_linear",
    model_key = "enet",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "enet_engineered",
    workflow_label = "Elastic Net - Engineered",
    model_type = "Elastic net",
    recipe_family = "Engineered",
    recipe_key = "engineered_linear",
    model_key = "enet",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "lasso_curated",
    workflow_label = "Lasso - Curated",
    model_type = "Lasso",
    recipe_family = "Curated",
    recipe_key = "curated_linear",
    model_key = "lasso",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "ridge_curated",
    workflow_label = "Ridge - Curated",
    model_type = "Ridge",
    recipe_family = "Curated",
    recipe_key = "curated_linear",
    model_key = "ridge",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "enet_curated",
    workflow_label = "Elastic Net - Curated",
    model_type = "Elastic net",
    recipe_family = "Curated",
    recipe_key = "curated_linear",
    model_key = "enet",
    tuned = TRUE
  )
)

# Assemble bundle
model_bundle_regularized <- list(
  workflow_info = workflow_info,
  specs = list(
    lasso = lasso_spec,
    ridge = ridge_spec,
    enet = enet_spec
  ),
  workflows = list(
    lasso_basic = workflow() |>
      add_model(lasso_spec) |>
      add_recipe(recipes$basic_linear),
    lasso_engineered = workflow() |>
      add_model(lasso_spec) |>
      add_recipe(recipes$engineered_linear),
    ridge_basic = workflow() |>
      add_model(ridge_spec) |>
      add_recipe(recipes$basic_linear),
    ridge_engineered = workflow() |>
      add_model(ridge_spec) |>
      add_recipe(recipes$engineered_linear),
    enet_basic = workflow() |>
      add_model(enet_spec) |>
      add_recipe(recipes$basic_linear),
    enet_engineered = workflow() |>
      add_model(enet_spec) |>
      add_recipe(recipes$engineered_linear),
    lasso_curated = workflow() |>
      add_model(lasso_spec) |>
      add_recipe(recipes$curated_linear),
    ridge_curated = workflow() |>
      add_model(ridge_spec) |>
      add_recipe(recipes$curated_linear),
    enet_curated = workflow() |>
      add_model(enet_spec) |>
      add_recipe(recipes$curated_linear)
  ),
  grids = list(
    lasso_basic = grid_regular(
      penalty(range = c(-4, -0.5)),
      levels = 25
    ),
    lasso_engineered = grid_regular(
      penalty(range = c(-4, -0.5)),
      levels = 25
    ),
    ridge_basic = grid_regular(
      penalty(range = c(-4, 0)),
      levels = 25
    ),
    ridge_engineered = grid_regular(
      penalty(range = c(-4, 0)),
      levels = 25
    ),
    enet_basic = grid_regular(
      penalty(range = c(-4, -0.5)),
      mixture(range = c(0, 1)),
      levels = c(9, 7)
    ),
    enet_engineered = grid_regular(
      penalty(range = c(-4, -0.5)),
      mixture(range = c(0, 1)),
      levels = c(9, 7)
    ),
    lasso_curated = grid_regular(
      penalty(range = c(-4, -0.5)),
      levels = 25
    ),
    ridge_curated = grid_regular(
      penalty(range = c(-4, 0)),
      levels = 25
    ),
    enet_curated = grid_regular(
      penalty(range = c(-4, -0.5)),
      mixture(range = c(0, 1)),
      levels = c(9, 7)
    )
  )
)
