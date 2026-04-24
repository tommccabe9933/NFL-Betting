# Fit baseline and logistic regression workflows

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "00_project_setup.R"))

prepared_objects <- read_rds(here("objects", "prepared_modeling_objects.rds"))
workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))

workflow_ids <- c("null_basic", "logistic_basic", "logistic_engineered", "logistic_curated")

workflow_results <- fit_workflow_set(workflow_ids, prepared_objects, workflow_objects)

for (name in names(workflow_results)) {
  write_rds(workflow_results[[name]], here("results", "workflows", paste0(name, ".rds")))
}
write_rds(workflow_results, here("objects", "workflow_results_baseline_logistic.rds"))
