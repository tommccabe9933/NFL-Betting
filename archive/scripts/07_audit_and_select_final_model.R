# Select final model and evaluate on test set

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "00_project_setup.R"))

prepared_objects <- read_rds(here("objects", "prepared_modeling_objects.rds"))
workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))
workflow_results <- read_rds(here("objects", "workflow_results.rds"))
workflow_info <- workflow_objects$workflow_info

official_comparison <- read_csv(here("results", "official_workflow_comparison.csv"), show_col_types = FALSE)

# Select model with highest CV ROC AUC
selected_row <- official_comparison |>
  arrange(desc(roc_auc), desc(accuracy)) |>
  slice_head(n = 1)

selected_id <- selected_row$workflow_id[[1]]
selected_workflow <- workflow_objects$workflows[[selected_id]]
selected_result <- workflow_results[[selected_id]]

# Fit selected model on full training data
selected_fit <- finalize_workflow_fit(
  workflow_object = selected_workflow,
  workflow_result = selected_result,
  train_data = prepared_objects$train_data
)

baseline_fit <- finalize_workflow_fit(
  workflow_object = workflow_objects$workflows[["null_basic"]],
  workflow_result = workflow_results[["null_basic"]],
  train_data = prepared_objects$train_data
)

# Generate test predictions
selected_probabilities <- predict(selected_fit$fit, prepared_objects$test_data, type = "prob")
selected_predictions <- predict(selected_fit$fit, prepared_objects$test_data, type = "class")
baseline_probabilities <- predict(baseline_fit$fit, prepared_objects$test_data, type = "prob")

best_predictions <- prepared_objects$test_data |>
  select(game_id_norm, season, week, fav_cover) |>
  bind_cols(selected_probabilities, selected_predictions)

baseline_predictions <- prepared_objects$test_data |>
  select(game_id_norm, season, week, fav_cover) |>
  bind_cols(baseline_probabilities) |>
  mutate(.pred_class = factor(ifelse(.pred_cover >= 0.5, "cover", "no_cover"), levels = c("cover", "no_cover")))

# Compute test metrics
test_metrics <- tribble(
  ~evaluation_set, ~workflow_label, ~roc_auc, ~accuracy, ~sensitivity, ~specificity,
  "Held-out test set", selected_row$workflow_label[[1]],
  roc_auc_vec(best_predictions$fav_cover, best_predictions$.pred_cover),
  accuracy_vec(best_predictions$fav_cover, best_predictions$.pred_class),
  sensitivity_vec(best_predictions$fav_cover, best_predictions$.pred_class),
  specificity_vec(best_predictions$fav_cover, best_predictions$.pred_class),
  "Held-out test set", "Baseline",
  roc_auc_vec(baseline_predictions$fav_cover, baseline_predictions$.pred_cover),
  accuracy_vec(baseline_predictions$fav_cover, baseline_predictions$.pred_class),
  sensitivity_vec(baseline_predictions$fav_cover, baseline_predictions$.pred_class),
  specificity_vec(baseline_predictions$fav_cover, baseline_predictions$.pred_class)
)

# Confusion matrix
confusion_table <- as.data.frame(
  conf_mat(best_predictions, truth = fav_cover, estimate = .pred_class)$table
) |>
  as_tibble() |>
  rename(`Observed outcome` = Truth, `Predicted cover` = Prediction, Count = Freq)

# Final selection record
final_selection <- tibble(
  workflow_id = selected_id,
  workflow_label = selected_row$workflow_label[[1]],
  recipe_family = selected_row$recipe_family[[1]],
  model_type = selected_row$model_type[[1]],
  training_roc_auc = selected_row$roc_auc[[1]],
  training_accuracy = selected_row$accuracy[[1]]
)

# Save outputs
write_csv(official_comparison, here("results", "all_workflow_comparison.csv"))
write_csv(final_selection, here("results", "final_selection.csv"))
write_csv(test_metrics, here("results", "test_metrics.csv"))
write_csv(confusion_table, here("results", "best_workflow_confusion_matrix.csv"))
write_rds(best_predictions, here("objects", "best_workflow_test_predictions.rds"))
