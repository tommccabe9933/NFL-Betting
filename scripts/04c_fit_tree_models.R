# Fit tree-based models (Random Forest + XGBoost)

library(tidyverse)
library(tidymodels)
library(here)
library(xgboost)

source(here("scripts", "00_project_setup.R"))

prepared_objects <- read_rds(here("objects", "prepared_modeling_objects.rds"))
workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))

all_tree_ids <- c("rf_basic", "rf_engineered", "xgb_basic", "xgb_engineered")

workflow_results <- fit_workflow_set(all_tree_ids, prepared_objects, workflow_objects)

for (name in names(workflow_results)) {
  write_rds(workflow_results[[name]], here("results", "workflows", paste0(name, ".rds")))
}
write_rds(workflow_results, here("objects", "workflow_results_tree_models.rds"))
