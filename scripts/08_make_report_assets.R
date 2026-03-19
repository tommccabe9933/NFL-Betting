# Generate figures and report assets

library(tidyverse)
library(tidymodels)
library(here)
library(broom)

source(here("scripts", "00_project_setup.R"))

official_comparison <- read_csv(here("results", "official_workflow_comparison.csv"), show_col_types = FALSE)
final_selection <- read_csv(here("results", "final_selection.csv"), show_col_types = FALSE)
workflow_results <- read_rds(here("objects", "workflow_results.rds"))

# Identify selected workflow for tuning plot
selected_id <- final_selection$workflow_id[[1]]

selected_is_tuned <- official_comparison |>
  filter(workflow_id == selected_id) |>
  pull(tuned) |>
  first()

# Save selected model tuning plot
dir.create(here("figures"), showWarnings = FALSE)

if (selected_is_tuned) {
  ggsave(
    filename = here("figures", "selected_tuning.png"),
    plot = autoplot(workflow_results[[selected_id]]) +
      theme_minimal(base_size = 12) +
      labs(title = paste("Tuning Results:", final_selection$workflow_label[[1]])),
    width = 8, height = 5, dpi = 300
  )
}

# Generate autoplot for every tuned workflow
tuned_ids <- official_comparison |> filter(tuned) |> pull(workflow_id)

walk(tuned_ids, function(wf_id) {
  p <- autoplot(workflow_results[[wf_id]]) +
    theme_minimal(base_size = 12) +
    labs(title = paste("Tuning Results:", wf_id))
  ggsave(filename = here("figures", paste0("tuning_", wf_id, ".png")),
         plot = p, width = 8, height = 5, dpi = 300)
})

#  MyROI simulator 
best_predictions <- read_rds(here("objects", "best_workflow_test_predictions.rds"))

set.seed(42)
n_games <- nrow(best_predictions)

roi_data <- best_predictions |>
  mutate(
    game_num = row_number(),
    actual_cover = as.integer(fav_cover == "cover"),
    model_pick = as.integer(.pred_cover > 0.5),
    coinflip_pick = sample(c(0L, 1L), n_games, replace = TRUE),
    always_fav_pick = 1L,
    model_profit = if_else(model_pick == actual_cover, 100, -110),
    coinflip_profit = if_else(coinflip_pick == actual_cover, 100, -110),
    always_fav_profit = if_else(always_fav_pick == actual_cover, 100, -110),
    model_cumulative = cumsum(model_profit),
    coinflip_cumulative = cumsum(coinflip_profit),
    always_fav_cumulative = cumsum(always_fav_profit)
  )

roi_long <- roi_data |>
  select(game_num, model_cumulative, coinflip_cumulative, always_fav_cumulative) |>
  pivot_longer(cols = ends_with("_cumulative"), names_to = "strategy", values_to = "cumulative_profit") |>
  mutate(strategy = case_when(
    strategy == "model_cumulative" ~ "Model Prediction",
    strategy == "coinflip_cumulative" ~ "Coin Flip (Random)",
    strategy == "always_fav_cumulative" ~ "Always Pick Favorite"
  ))

roi_plot <- ggplot(roi_long, aes(x = game_num, y = cumulative_profit, color = strategy)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c(
    "Model Prediction" = "#013369",
    "Coin Flip (Random)" = "#999999",
    "Always Pick Favorite" = "#D50A0A"
  )) +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Simulated Betting ROI on Held-Out Test Games",
    subtitle = "Each bet risks $110 to win $100 (standard -110 odds)",
    x = "Game number (test set)",
    y = "Cumulative profit / loss",
    color = "Strategy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "grey40"),
    legend.position = "bottom"
  )

ggsave(filename = here("figures", "roi_simulation.png"), plot = roi_plot, width = 8, height = 5, dpi = 300)

# Coefficient table for selected model
prepared_objects <- read_rds(here("objects", "prepared_modeling_objects.rds"))
workflow_objects <- read_rds(here("objects", "workflow_objects.rds"))

selected_wf <- workflow_objects$workflows[[selected_id]]
selected_result <- workflow_results[[selected_id]]

if (selected_is_tuned) {
  best_params <- select_best(selected_result, metric = "roc_auc")
  selected_wf <- finalize_workflow(selected_wf, best_params)
}

final_fit <- fit(selected_wf, data = prepared_objects$train_data)

# Creating clean labels for all these variables

clean_labels <- c(
  "fav_roll5_pts"        = "Favorite Rolling 5-Gm Points",
  "pts_play_diff"        = "Points Per Play Diff",
  "fav_roll5_ypp"        = "Favorite Rolling 5-Gm Yards/Play",
  "ypp_diff"             = "Yards Per Play Diff",
  "turnover_diff"        = "Turnover Differential",
  "off_epa_diff"         = "Offensive EPA Diff",
  "weather_temperature"  = "Temperature",
  "dog_short_week_yes"   = "Underdog on Short Week",
  "to_momentum_diff"     = "Turnover Momentum Diff",
  "elo_diff"             = "Elo Rating Diff",
  "def_epa_diff"         = "Defensive EPA Diff",
  "success_rate_diff"    = "Success Rate Diff",
  "stadium_neutral_TRUE."= "Neutral Site",
  "rush_eff_diff"        = "Rush Efficiency Diff",
  "fav_short_week_yes"   = "Favorite on Short Week",
  "sack_rate_diff"       = "Sack Rate Diff",
  "rest_diff"            = "Rest Differential",
  "third_down_diff"      = "Third Down Conv. Diff",
  "over_under_line"      = "Over/Under Line",
  "fav_at_home_yes"      = "Favorite at Home",
  "spread_fav"           = "Point Spread",
  "momentum_diff"        = "Momentum Diff (5-gm vs Season)",
  "pass_eff_diff"        = "Pass Efficiency Diff",
  "weather_wind_mph"     = "Wind Speed",
  "dog_roll5_ypp"        = "Underdog Rolling 5-Gm Yards/Play",
  "dog_roll5_pts"        = "Underdog Rolling 5-Gm Points"
)

coef_data <- tidy(final_fit) |>
  filter(term != "(Intercept)", !is.na(estimate)) |>
  mutate(
    label = coalesce(clean_labels[term], term),
    direction = if_else(estimate > 0, "Increases Cover Probability", "Decreases Cover Probability")
  ) |>
  arrange(desc(abs(estimate)))

write_csv(coef_data, here("results", "model_coefficients.csv"))
