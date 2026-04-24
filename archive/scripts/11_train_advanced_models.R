# Train models with advanced features from script 10
# Tests multiple recipes: original curated, advanced curated, kitchen sink v2

library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(conflicted)

conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
tidymodels_prefer()

available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

cat("Loading data...\n")
model_data_v2 <- read_rds(here("objects", "modeling_data_v2.rds"))

cat("Dataset:", nrow(model_data_v2), "rows x", ncol(model_data_v2), "cols\n")

# Temporal split
train_data <- model_data_v2 |> filter(season <= 2020)
test_data <- model_data_v2 |> filter(season >= 2021)

cat("Train:", nrow(train_data), "| Test:", nrow(test_data), "\n")

# 10-fold CV, 3 repeats
set.seed(301)
cv_folds <- vfold_cv(train_data, v = 10, repeats = 3, strata = fav_cover)

# ============================================================
# RECIPE 1: Original curated (baseline comparison)
# ============================================================
curated_vars_original <- c(
  "fav_cover", "game_id_norm", "season", "week",
  "fav_roll5_ypp", "ypp_diff", "turnover_diff", "dog_roll5_pts",
  "weather_temperature", "off_epa_diff", "dog_short_week",
  "elo_diff", "def_epa_diff", "to_momentum_diff", "success_rate_diff"
)

recipe_original <- recipe(fav_cover ~ ., data = train_data |> select(any_of(curated_vars_original))) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# ============================================================
# RECIPE 2: Advanced curated (new high-signal features)
# ============================================================
curated_vars_advanced <- c(
  "fav_cover", "game_id_norm", "season", "week",
  # Original strong predictors (keep what works)
  "spread_fav", "elo_diff", "ypp_diff", "turnover_diff",
  "fav_roll5_ypp", "dog_roll5_pts", "weather_temperature",
  # QB quality (new)
  "qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
  # Situational EPA (new - the market may not fully price these)
  "third_down_epa_diff", "redzone_epa_diff",
  "pass_epa_mismatch", "rush_epa_mismatch",
  "explosive_diff", "neg_play_diff",
  # Injury impact (new)
  "injury_diff", "fav_qb_out", "dog_qb_out",
  # Context (new)
  "div_game", "is_primetime", "is_dome",
  # Rest
  "dog_short_week"
)

recipe_advanced <- recipe(fav_cover ~ ., data = train_data |> select(any_of(curated_vars_advanced))) |>
  update_role(game_id_norm, season, week, new_role = "id variable") |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# ============================================================
# RECIPE 3: Kitchen sink v2 (all original + all new features)
# ============================================================
# Drop the raw fav/dog columns that are just inputs to differentials
id_vars <- c("game_id_norm", "season", "week", "team_fav", "team_dog")

recipe_kitchensink_v2 <- recipe(fav_cover ~ ., data = train_data |> select(-any_of(id_vars))) |>
  step_zv(all_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# ============================================================
# MODEL SPECS
# ============================================================
# Ridge (our current best)
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("classification")

# Lasso (feature selection built in)
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("classification")

# Elastic net
enet_spec <- logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("classification")

# XGBoost (to test nonlinearity)
xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  sample_size = tune()
) |>
  set_engine("xgboost") |>
  set_mode("classification")

# Tuning grids
ridge_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 25)
lasso_grid <- grid_regular(penalty(range = c(-4, -0.5)), levels = 25)
enet_grid <- grid_regular(penalty(range = c(-4, -0.5)), mixture(range = c(0, 1)), levels = c(10, 5))
xgb_grid <- grid_latin_hypercube(
  tree_depth(range = c(1, 6)),
  learn_rate(range = c(-3, -0.5)),
  min_n(range = c(5, 50)),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  size = 30
)

# ============================================================
# BUILD WORKFLOWS
# ============================================================
cat("\nBuilding workflows...\n")

workflows_list <- list(
  # Ridge across all 3 recipes
  ridge_original = workflow() |> add_model(ridge_spec) |> add_recipe(recipe_original),
  ridge_advanced = workflow() |> add_model(ridge_spec) |> add_recipe(recipe_advanced),
  ridge_kitchensink = workflow() |> add_model(ridge_spec) |> add_recipe(recipe_kitchensink_v2),
  # Lasso (good for feature selection with many features)
  lasso_advanced = workflow() |> add_model(lasso_spec) |> add_recipe(recipe_advanced),
  lasso_kitchensink = workflow() |> add_model(lasso_spec) |> add_recipe(recipe_kitchensink_v2),
  # Elastic net
  enet_advanced = workflow() |> add_model(enet_spec) |> add_recipe(recipe_advanced),
  # XGBoost (nonlinear - test on advanced recipe)
  xgb_advanced = workflow() |> add_model(xgb_spec) |> add_recipe(recipe_advanced)
)

grids_list <- list(
  ridge_original = ridge_grid,
  ridge_advanced = ridge_grid,
  ridge_kitchensink = ridge_grid,
  lasso_advanced = lasso_grid,
  lasso_kitchensink = lasso_grid,
  enet_advanced = enet_grid,
  xgb_advanced = xgb_grid
)

# ============================================================
# TUNE ALL WORKFLOWS
# ============================================================
cat("Tuning", length(workflows_list), "workflows...\n\n")

results_list <- list()

for (wf_name in names(workflows_list)) {
  cat("  Tuning:", wf_name, "...\n")

  set.seed(301)
  result <- tune_grid(
    workflows_list[[wf_name]],
    resamples = cv_folds,
    grid = grids_list[[wf_name]],
    metrics = metric_set(roc_auc, accuracy),
    control = control_grid(save_pred = FALSE, verbose = FALSE)
  )

  best <- select_best(result, metric = "roc_auc")
  best_metrics <- collect_metrics(result) |>
    semi_join(best, by = intersect(names(best), names(collect_metrics(result)))) |>
    filter(.metric == "roc_auc")

  cat("    Best ROC AUC:", round(best_metrics$mean, 4), "\n")

  results_list[[wf_name]] <- list(
    tune_result = result,
    best_params = best,
    cv_roc_auc = best_metrics$mean,
    cv_accuracy = (collect_metrics(result) |>
      semi_join(best, by = intersect(names(best), names(collect_metrics(result)))) |>
      filter(.metric == "accuracy"))$mean
  )
}

# ============================================================
# COMPARISON TABLE
# ============================================================
cat("\n=== CROSS-VALIDATION RESULTS ===\n")

comparison <- map_dfr(names(results_list), function(wf_name) {
  r <- results_list[[wf_name]]
  tibble(
    workflow = wf_name,
    cv_roc_auc = round(r$cv_roc_auc, 4),
    cv_accuracy = round(r$cv_accuracy, 4)
  )
}) |>
  arrange(desc(cv_roc_auc))

print(comparison)
write_csv(comparison, here("results", "advanced_model_comparison.csv"))

# ============================================================
# FIT BEST MODEL ON FULL TRAINING DATA AND PREDICT TEST SET
# ============================================================
best_wf_name <- comparison$workflow[1]
cat("\nBest workflow:", best_wf_name, "\n")

best_result <- results_list[[best_wf_name]]
best_wf <- finalize_workflow(workflows_list[[best_wf_name]], best_result$best_params)
best_fit <- fit(best_wf, data = train_data)

# Test predictions
test_probs <- predict(best_fit, new_data = test_data, type = "prob")
test_class <- predict(best_fit, new_data = test_data, type = "class")

test_preds <- test_data |>
  select(game_id_norm, season, week, fav_cover) |>
  bind_cols(test_probs, test_class)

# Test metrics
test_accuracy <- accuracy_vec(test_preds$fav_cover, test_preds$.pred_class)
test_roc <- roc_auc_vec(test_preds$fav_cover, test_preds$.pred_cover, event_level = "first")

cat("Test ROC AUC:", round(test_roc, 4), "\n")
cat("Test Accuracy:", round(test_accuracy, 4), "\n")

# ROI simulation
roi <- test_preds |>
  mutate(
    actual_cover = as.integer(fav_cover == "cover"),
    model_pick = as.integer(.pred_cover > 0.5),
    correct = model_pick == actual_cover,
    profit = if_else(correct, 100, -110)
  )

cat("Win rate:", round(mean(roi$correct) * 100, 1), "%\n")
cat("Total profit:", scales::dollar(sum(roi$profit)), "\n")
cat("ROI:", round(sum(roi$profit) / (nrow(roi) * 110) * 100, 1), "%\n")

# Season breakdown
cat("\nSeason breakdown:\n")
roi |>
  group_by(season) |>
  summarize(
    n = n(),
    accuracy = round(mean(correct), 3),
    profit = sum(profit),
    roi_pct = round(sum(profit) / (n() * 110) * 100, 1),
    .groups = "drop"
  ) |>
  print()

# Save
write_rds(test_preds, here("objects", "advanced_model_test_predictions.rds"))
write_rds(results_list, here("objects", "advanced_model_results.rds"))

cat("\nDone! Saved to objects/advanced_model_test_predictions.rds\n")
