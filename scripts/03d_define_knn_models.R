# KNN model spec

knn_spec <- nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")

# Define workflow metadata (one row per workflow)
workflow_info <- bind_rows(
  tibble(
    workflow_id = "knn_basic",
    workflow_label = "K-Nearest Neighbors - Basic",
    model_type = "K-nearest neighbors",
    recipe_family = "Basic",
    recipe_key = "basic_linear",
    model_key = "knn",
    tuned = TRUE
  ),
  tibble(
    workflow_id = "knn_engineered",
    workflow_label = "K-Nearest Neighbors - Engineered",
    model_type = "K-nearest neighbors",
    recipe_family = "Engineered",
    recipe_key = "engineered_linear",
    model_key = "knn",
    tuned = TRUE
  )
)

# Assemble bundle
model_bundle_knn <- list(
  workflow_info = workflow_info,
  specs = list(
    knn = knn_spec
  ),
  workflows = list(
    knn_basic = workflow() |>
      add_model(knn_spec) |>
      add_recipe(recipes$basic_linear),
    knn_engineered = workflow() |>
      add_model(knn_spec) |>
      add_recipe(recipes$engineered_linear)
  ),
  grids = list(
    knn_basic = grid_regular(
      neighbors(range = c(15L, 501L)),
      levels = 20
    ),
    knn_engineered = grid_regular(
      neighbors(range = c(15L, 501L)),
      levels = 20
    )
  )
)
