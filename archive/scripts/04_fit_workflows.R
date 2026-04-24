# Master fitting script

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "04a_fit_baseline_logistic.R"))
source(here("scripts", "04b_fit_regularized_models.R"))
source(here("scripts", "04c_fit_tree_models.R"))
source(here("scripts", "04d_fit_knn_models.R"))

workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))
workflow_info <- workflow_objects$workflow_info

# Combine all partial results into a single list
workflow_results <- c(
  read_rds(here("objects", "workflow_results_baseline_logistic.rds")),
  read_rds(here("objects", "workflow_results_regularized.rds")),
  read_rds(here("objects", "workflow_results_tree_models.rds")),
  read_rds(here("objects", "workflow_results_knn.rds"))
)

workflow_results <- workflow_results[workflow_info$workflow_id]

write_rds(workflow_results, here("objects", "workflow_results.rds"))
