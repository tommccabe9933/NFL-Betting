# I am defining all workflows, metrics, and control objects

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "00_project_setup.R"))

prepared_objects <- read_rds(here("objects", "prepared_modeling_objects.rds"))
train_data <- prepared_objects$train_data
recipes <- prepared_objects$recipes

metric_set_used <- metric_set(roc_auc, accuracy)

control_grid_used <- control_grid(
  save_pred = TRUE,
  save_workflow = TRUE,
  parallel_over = "everything"
)

control_resamples_used <- control_resamples(
  save_pred = TRUE,
  save_workflow = TRUE,
  parallel_over = "everything"
)

predictor_count <- train_data |>
  select(-fav_cover) |>
  ncol()

# Source model definition scripts
source(here("scripts", "03a_define_baseline_logistic.R"))
source(here("scripts", "03b_define_regularized_models.R"))
source(here("scripts", "03c_define_tree_models.R"))
source(here("scripts", "03d_define_knn_models.R"))

workflow_objects <- combine_model_bundles(
  bundles = list(
    model_bundle_baseline_logistic,
    model_bundle_regularized,
    model_bundle_tree,
    model_bundle_knn
  ),
  metrics = metric_set_used,
  control_grid = control_grid_used,
  control_resamples = control_resamples_used
)

write_rds(workflow_objects, here("objects", "workflow_objects.rds"))
write_csv(workflow_objects$workflow_info, here("results", "workflow_reference.csv"))
