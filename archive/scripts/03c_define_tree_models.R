# Tree-based model specs: Random Forest and XGBoost

rf_spec <- rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) |>
  set_engine("xgboost") |>
  set_mode("classification")

# Define workflow metadata (one row per workflow)
workflow_info <- bind_rows(
  tibble(
    workflow_id = "rf_basic",
    workflow_label = "Random Forest - Basic",
    model_type = "Random forest",
    recipe_family = "Basic",
    recipe_key = "basic_tree",
    model_key = "rf",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "rf_engineered",
    workflow_label = "Random Forest - Engineered",
    model_type = "Random forest",
    recipe_family = "Engineered",
    recipe_key = "engineered_tree",
    model_key = "rf",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "xgb_basic",
    workflow_label = "XGBoost - Basic",
    model_type = "Gradient boosting",
    recipe_family = "Basic",
    recipe_key = "basic_tree",
    model_key = "xgb",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "xgb_engineered",
    workflow_label = "XGBoost - Engineered",
    model_type = "Gradient boosting",
    recipe_family = "Engineered",
    recipe_key = "engineered_tree",
    model_key = "xgb",
    tuned = TRUE
  )
)

# Assemble bundle
model_bundle_tree <- list(
  workflow_info = workflow_info,
  specs = list(
    rf = rf_spec,
    xgb = xgb_spec
  ),
  workflows = list(
    rf_basic = workflow() |>
      add_model(rf_spec) |>
      add_recipe(recipes$basic_tree),
    rf_engineered = workflow() |>
      add_model(rf_spec) |>
      add_recipe(recipes$engineered_tree),
    xgb_basic = workflow() |>
      add_model(xgb_spec) |>
      add_recipe(recipes$basic_tree),
    xgb_engineered = workflow() |>
      add_model(xgb_spec) |>
      add_recipe(recipes$engineered_tree)
  ),
  grids = list(
    rf_basic = grid_regular(
      finalize(mtry(range = c(2L, min(18L, predictor_count))), train_data),
      min_n(range = c(3L, 80L)),
      levels = c(6, 5)
    ),
    rf_engineered = grid_regular(
      finalize(mtry(range = c(2L, min(18L, predictor_count))), train_data),
      min_n(range = c(3L, 80L)),
      levels = c(6, 5)
    ),
    xgb_basic = grid_regular(
      trees(range = c(250L, 750L)),
      tree_depth(range = c(2L, 8L)),
      learn_rate(range = c(-3, 0)),
      min_n(range = c(3L, 80L)),
      loss_reduction(range = c(-3, 1)),
      sample_prop(range = c(0.5, 1.0)),
      levels = c(
        trees = 3,
        tree_depth = 3,
        learn_rate = 3,
        min_n = 3,
        loss_reduction = 2,
        sample_size = 2
      )
    ),
    xgb_engineered = grid_regular(
      trees(range = c(250L, 750L)),
      tree_depth(range = c(2L, 8L)),
      learn_rate(range = c(-3, 0)),
      min_n(range = c(3L, 80L)),
      loss_reduction(range = c(-3, 1)),
      sample_prop(range = c(0.5, 1.0)),
      levels = c(
        trees = 3,
        tree_depth = 3,
        learn_rate = 3,
        min_n = 3,
        loss_reduction = 2,
        sample_size = 2
      )
    )
  )
)
