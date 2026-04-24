# Diagnose temporal decay in the static model's test predictions

library(tidyverse)
library(tidymodels)
library(here)

best_predictions <- read_rds(here("objects", "best_workflow_test_predictions.rds"))

# Add game numbering and profit calculation
roi_data <- best_predictions |>
  mutate(
    game_num = row_number(),
    actual_cover = as.integer(fav_cover == "cover"),
    model_pick = as.integer(.pred_cover > 0.5),
    model_correct = model_pick == actual_cover,
    model_profit = if_else(model_correct, 100, -110)
  )

# Helper to compute slice metrics
compute_slice_metrics <- function(df, label) {
  tibble(
    slice = label,
    n_games = nrow(df),
    accuracy = mean(df$model_correct),
    win_rate_pct = mean(df$model_correct) * 100,
    roc_auc = tryCatch(
      roc_auc_vec(df$fav_cover, df$.pred_cover, event_level = "first"),
      error = function(e) NA_real_
    ),
    total_profit = sum(df$model_profit),
    roi_pct = sum(df$model_profit) / (nrow(df) * 110) * 100
  )
}

# Slice: first 400 vs rest
slice_400 <- compute_slice_metrics(roi_data |> filter(game_num <= 400), "First 400 games")
slice_rest <- compute_slice_metrics(roi_data |> filter(game_num > 400), "Games 401-1316")
slice_all <- compute_slice_metrics(roi_data, "All test games")

decay_analysis <- bind_rows(slice_400, slice_rest, slice_all)
write_csv(decay_analysis, here("results", "decay_analysis.csv"))

cat("Decay analysis:\n")
print(decay_analysis)

# Where does game 400 fall?
game_400_info <- roi_data |> filter(game_num == 400) |> select(game_num, season, week)
cat("\nGame 400 falls at:", paste("Season", game_400_info$season, "Week", game_400_info$week), "\n")

# Season-by-season breakdown
season_metrics <- roi_data |>
  group_by(season) |>
  group_modify(~ compute_slice_metrics(.x, as.character(.y$season))) |>
  ungroup() |>
  mutate(slice = paste("Season", season)) |>
  select(-season)

write_csv(season_metrics, here("results", "decay_season_metrics.csv"))

cat("\nSeason-by-season metrics:\n")
print(season_metrics)

# Find season boundary game numbers for annotation
season_boundaries <- roi_data |>
  group_by(season) |>
  summarize(
    first_game = min(game_num),
    last_game = max(game_num),
    n_games = n(),
    .groups = "drop"
  )

cat("\nSeason boundaries:\n")
print(season_boundaries)

write_rds(
  list(roi_data = roi_data, season_boundaries = season_boundaries),
  here("objects", "decay_diagnosis.rds")
)

cat("\nDone. Outputs: results/decay_analysis.csv, results/decay_season_metrics.csv, objects/decay_diagnosis.rds\n")
