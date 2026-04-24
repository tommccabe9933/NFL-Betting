# Compare static vs expanding window approaches

library(tidyverse)
library(tidymodels)
library(here)

dir.create(here("figures"), showWarnings = FALSE)

# Load predictions
static_preds <- read_rds(here("objects", "best_workflow_test_predictions.rds")) |>
  mutate(approach = "Static (Train 2012-2020)")

expanding_preds <- read_rds(here("objects", "expanding_window_predictions.rds")) |>
  mutate(approach = "Expanding Window")

# Load decay diagnosis for season boundaries
decay_obj <- read_rds(here("objects", "decay_diagnosis.rds"))
season_boundaries <- decay_obj$season_boundaries

# Unify column names and combine
static_clean <- static_preds |>
  select(game_id_norm, season, week, fav_cover, .pred_cover, .pred_class, approach)

expanding_clean <- expanding_preds |>
  select(game_id_norm, season, week, fav_cover, .pred_cover, .pred_class, approach)

all_preds <- bind_rows(static_clean, expanding_clean)

# Helper for metrics
compute_approach_metrics <- function(df, label) {
  actual_cover <- as.integer(df$fav_cover == "cover")
  model_pick <- as.integer(df$.pred_cover > 0.5)
  model_correct <- model_pick == actual_cover
  model_profit <- if_else(model_correct, 100L, -110L)

  tibble(
    label = label,
    n_games = nrow(df),
    accuracy = mean(model_correct),
    roc_auc = tryCatch(
      roc_auc_vec(df$fav_cover, df$.pred_cover, event_level = "first"),
      error = function(e) NA_real_
    ),
    total_profit = sum(model_profit),
    roi_pct = sum(model_profit) / (nrow(df) * 110) * 100,
    win_rate_pct = mean(model_correct) * 100
  )
}

# Season-by-season comparison
season_comparison <- all_preds |>
  group_by(approach, season) |>
  group_modify(~ compute_approach_metrics(.x, paste(.y$approach, .y$season))) |>
  ungroup() |>
  arrange(season, approach)

write_csv(season_comparison, here("results", "rolling_comparison.csv"))
cat("Season comparison:\n")
print(season_comparison |> select(approach, season, accuracy, roc_auc, total_profit, roi_pct))

# Overall comparison
overall_comparison <- all_preds |>
  group_by(approach) |>
  group_modify(~ compute_approach_metrics(.x, .y$approach)) |>
  ungroup()

write_csv(overall_comparison, here("results", "rolling_overall_comparison.csv"))
cat("\nOverall comparison:\n")
print(overall_comparison)

# Build cumulative profit for both approaches + baselines
# IMPORTANT: Use the natural row order from the RDS file (same as script 08)
# Do NOT re-sort by season/week — the original profitable early stretch
# depends on the natural data ordering.

# Static model: preserve original row order (matches script 08 ROI chart)
static_cum <- static_clean |>
  mutate(
    game_num = row_number(),
    actual_cover = as.integer(fav_cover == "cover"),
    model_pick = as.integer(.pred_cover > 0.5),
    model_profit = if_else(model_pick == actual_cover, 100, -110),
    cumulative_profit = cumsum(model_profit),
    strategy = "Static Model"
  ) |>
  select(game_num, season, cumulative_profit, strategy)

# Expanding window: match predictions to the same game order as static
# Join on game_id_norm to get the same row ordering
expanding_ordered <- static_clean |>
  mutate(game_num = row_number()) |>
  select(game_id_norm, game_num) |>
  inner_join(expanding_clean, by = "game_id_norm")

expanding_cum <- expanding_ordered |>
  arrange(game_num) |>
  mutate(
    actual_cover = as.integer(fav_cover == "cover"),
    model_pick = as.integer(.pred_cover > 0.5),
    model_profit = if_else(model_pick == actual_cover, 100, -110),
    cumulative_profit = cumsum(model_profit),
    strategy = "Expanding Window"
  ) |>
  select(game_num, season, cumulative_profit, strategy)

# Baselines (use static predictions for game ordering — same natural order)
set.seed(42)
n_games <- nrow(static_clean)
baseline_data <- static_clean |>
  mutate(
    game_num = row_number(),
    actual_cover = as.integer(fav_cover == "cover"),
    coinflip_pick = sample(c(0L, 1L), n_games, replace = TRUE),
    coinflip_profit = if_else(coinflip_pick == actual_cover, 100, -110),
    always_fav_profit = if_else(1L == actual_cover, 100, -110)
  )

coinflip_cum <- baseline_data |>
  mutate(cumulative_profit = cumsum(coinflip_profit), strategy = "Coin Flip") |>
  select(game_num, season, cumulative_profit, strategy)

always_fav_cum <- baseline_data |>
  mutate(cumulative_profit = cumsum(always_fav_profit), strategy = "Always Pick Favorite") |>
  select(game_num, season, cumulative_profit, strategy)

all_cum <- bind_rows(static_cum, expanding_cum, coinflip_cum, always_fav_cum)

# Season boundary lines for annotation
boundary_lines <- season_boundaries |>
  filter(season > min(season)) |>
  pull(first_game)

# Centerpiece figure: cumulative profit comparison
roi_plot <- ggplot(all_cum, aes(x = game_num, y = cumulative_profit, color = strategy)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = boundary_lines, linetype = "dotted", color = "grey70", linewidth = 0.4) +
  scale_color_manual(values = c(
    "Expanding Window" = "#013369",
    "Static Model" = "#999999",
    "Coin Flip" = "#BBBBBB",
    "Always Pick Favorite" = "#D50A0A"
  )) +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Static Model vs. Expanding Window Retraining",
    subtitle = "Cumulative profit on held-out test games ($110 risk / $100 win per bet)",
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

ggsave(here("figures", "rolling_vs_static_roi.png"), roi_plot, width = 10, height = 6, dpi = 300)

# Decay annotated figure (static model only with game 400 line)
static_annotated <- static_cum |>
  bind_rows(coinflip_cum, always_fav_cum)

decay_plot <- ggplot(static_annotated, aes(x = game_num, y = cumulative_profit, color = strategy)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 400, linetype = "dashed", color = "#e76f51", linewidth = 0.8) +
  geom_vline(xintercept = boundary_lines, linetype = "dotted", color = "grey70", linewidth = 0.4) +
  annotate("text", x = 415, y = 2500, label = "Model decay\nbegins (~game 400)",
           color = "#e76f51", hjust = 0, size = 3.5, fontface = "italic") +
  scale_color_manual(values = c(
    "Static Model" = "#013369",
    "Coin Flip" = "#999999",
    "Always Pick Favorite" = "#D50A0A"
  )) +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Temporal Decay in the Static Model",
    subtitle = "Model profitable early, then collapses as training data grows stale",
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

ggsave(here("figures", "decay_annotated.png"), decay_plot, width = 10, height = 6, dpi = 300)

cat("\nDone. Figures: figures/rolling_vs_static_roi.png, figures/decay_annotated.png\n")
