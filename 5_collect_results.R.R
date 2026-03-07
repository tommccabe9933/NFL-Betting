library(tidyverse)
library(tidymodels)

null_metrics <- collect_metrics(null_results) |>
  mutate(model = "Null")

log_metrics <- collect_metrics(log_results) |>
  mutate(model = "Logistic Regression")

model_results <- bind_rows(null_metrics, log_metrics)

model_results |>
  select(model, .metric, mean) |>
  pivot_wider(
    names_from = .metric,
    values_from = mean
  )