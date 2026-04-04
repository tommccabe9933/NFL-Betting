# Collect and compare all workflow results

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "00_project_setup.R"))

workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))
workflow_results <- read_rds(here("objects", "workflow_results.rds"))
workflow_info <- workflow_objects$workflow_info

# Build comparison table across all official workflows
workflow_comparison <- map_dfr(
  workflow_info$workflow_id,
  function(current_id) {
    workflow_row <- filter(workflow_info, workflow_id == current_id)
    extract_workflow_summary(workflow_row, workflow_results[[current_id]])
  }
) |>
  mutate(comparison_stage = "official") |>
  arrange(desc(roc_auc), desc(accuracy))

# Building a function to make all tuning metrics for appendix plots
tuning_parameter_archive <- map_dfr(
  workflow_info$workflow_id[workflow_info$tuned],
  function(current_id) {
    workflow_row <- filter(workflow_info, workflow_id == current_id)
    collect_metrics(workflow_results[[current_id]]) |>
      mutate(
        workflow_id = current_id,
        workflow_label = workflow_row$workflow_label,
        recipe_family = workflow_row$recipe_family
      )
  }
)

write_csv(workflow_comparison, here("results", "official_workflow_comparison.csv"))
write_csv(workflow_comparison, here("results", "workflow_comparison.csv"))
write_csv(tuning_parameter_archive, here("results", "tuning_parameter_archive.csv"))
