# Fit expanded/refined workflows for close contenders

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "00_project_setup.R"))

prepared_objects <- read_rds(here("objects", "prepared_modeling_objects.rds"))
workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))
official_comparison <- read_csv(here("results", "official_workflow_comparison.csv"), show_col_types = FALSE)

workflow_info <- workflow_objects$workflow_info
recipes <- prepared_objects$recipes
cv_folds <- prepared_objects$cv_folds

official_best <- official_comparison |>
  filter(comparison_stage == "official") |>
  slice_max(roc_auc, with_ties = FALSE)



# This was a fun Stat attempt where I identified close contenders within 1 SE of the best

close_finalists <- official_comparison |>
  inner_join(
    select(workflow_info, workflow_id, recipe_key, model_key, workflow_tuned = tuned),
    by = "workflow_id"
  ) |>
  filter(workflow_tuned, workflow_id != official_best$workflow_id[[1]]) |>
  mutate(
    gap = official_best$roc_auc[[1]] - roc_auc,
    cutoff = max(official_best$roc_auc_std_err[[1]], 1e-4)
  ) |>
  filter(gap <= cutoff) |>
  arrange(gap, desc(roc_auc)) |>
  slice_head(n = 4)


# Refined grids for each model type
expanded_grids_lookup <- list(
  knn   = grid_regular(neighbors(range = c(15L, 301L)), levels = 16),
  enet  = grid_regular(penalty(range = c(-4, -0.2)), mixture(range = c(0, 1)), levels = c(11, 5)),
  lasso = grid_regular(penalty(range = c(-4, -0.2)), levels = 20),
  ridge = grid_regular(penalty(range = c(-4, 0)), levels = 20),
  rf    = grid_regular(
    finalize(mtry(range = c(2L, min(20L, ncol(prepared_objects$train_data) - 1))), prepared_objects$train_data),
    min_n(range = c(3L, 60L)),
    levels = c(5, 5)
  ),
  
# Using grid_space_filling here instead

  xgb = grid_space_filling(
    trees(range = c(250L, 750L)),
    tree_depth(range = c(2L, 8L)),
    learn_rate(range = c(-3, 0)),
    min_n(range = c(3L, 80L)),
    loss_reduction(range = c(-3, 1)),
    sample_prop(range = c(0.5, 1.0)),
    size = 30
  )
)

expanded_entries <- close_finalists |>
  mutate(grid = map(model_key, ~ expanded_grids_lookup[[.x]])) |>
  filter(!map_lgl(grid, is.null)) |>
  mutate(
    workflow_id = paste0(workflow_id, "_refined"),
    workflow_label = paste0(workflow_label, " (Refined)")
  )

if (nrow(expanded_entries) == 0) {
  write_csv(tibble(), here("results", "expanded_workflow_comparison.csv"))
  write_rds(list(), here("objects", "expanded_workflow_results.rds"))
  write_rds(
    list(workflow_info = tibble(), workflows = list(), grids = list()),
    here("objects", "expanded_workflow_objects.rds")
  )
  write_csv(
    tibble(official_winner = official_comparison$workflow_label[[1]],
           official_winner_roc_auc = official_comparison$roc_auc[[1]]),
    here("results", "official_winner_snapshot.csv")
  )
  quit(status = 0)
}

# Building expanded workflows
expanded_workflows <- set_names(
  map(expanded_entries$workflow_id, function(target_id) {
    row <- expanded_entries |> filter(workflow_id == target_id) |> slice(1)
    spec <- workflow_objects$specs[[row$model_key]]
    rec <- recipes[[row$recipe_key]]
    workflow() |> add_model(spec) |> add_recipe(rec)
  }),
  expanded_entries$workflow_id
)

expanded_grids <- set_names(expanded_entries$grid, expanded_entries$workflow_id)

# Fitting expanded workflows
set.seed(302)
expanded_results <- imap(
  expanded_workflows,
  ~ tune_grid(
    .x,
    resamples = cv_folds,
    grid = expanded_grids[[.y]],
    metrics = workflow_objects$metrics,
    control = workflow_objects$control_grid
  )
)

expanded_workflow_info <- expanded_entries |>
  select(workflow_id, workflow_label, model_type, recipe_family, model_key, recipe_key) |>
  mutate(tuned = TRUE)

expanded_comparison <- map_dfr(
  expanded_workflow_info$workflow_id,
  function(current_id) {
    workflow_row <- filter(expanded_workflow_info, workflow_id == current_id)
    extract_workflow_summary(workflow_row, expanded_results[[current_id]])
  }
) |>
  mutate(comparison_stage = "expanded") |>
  arrange(desc(roc_auc))

# Save results
write_rds(expanded_results, here("objects", "expanded_workflow_results.rds"))
write_rds(
  list(workflow_info = expanded_workflow_info, workflows = expanded_workflows, grids = expanded_grids),
  here("objects", "expanded_workflow_objects.rds")
)
write_csv(expanded_comparison, here("results", "expanded_workflow_comparison.csv"))
write_csv(
  tibble(official_winner = official_comparison$workflow_label[[1]],
         official_winner_roc_auc = official_comparison$roc_auc[[1]]),
  here("results", "official_winner_snapshot.csv")
)
