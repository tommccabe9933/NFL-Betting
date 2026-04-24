# Feature Block Validation
# Tests whether each feature block (FTN, NGS, PFR) adds INCREMENTAL signal
# Uses: out-of-sample delta in log loss, bootstrap stability, permutation nulls

library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(conflicted)

conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
tidymodels_prefer()

available_cores <- parallel::detectCores(logical = TRUE) - 1
plan(multisession, workers = available_cores)

source(here("scripts", "14_research_protocol.R"))

model_data_v2 <- read_rds(here("objects", "modeling_data_v2.rds"))

# ============================================================
# FEATURE BLOCKS
# ============================================================

base_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  "spread_fav", "over_under_line", "elo_diff",
  "ypp_diff", "turnover_diff", "fav_roll5_ypp", "dog_roll5_pts",
  "weather_temperature", "to_momentum_diff", "success_rate_diff",
  "qb_epa_diff", "qb_cpoe_diff", "off_epa_diff", "def_epa_diff",
  "third_down_epa_diff", "redzone_epa_diff", "explosive_diff",
  "injury_diff", "div_game", "is_primetime",
  "rest_diff", "fav_at_home", "dog_short_week"
)

ftn_block <- c(
  "fav_ftn_roll5_blitz", "fav_ftn_roll5_rpo",
  "fav_ftn_roll5_play_action", "fav_ftn_roll5_pass_rush",
  "dog_ftn_roll5_blitz", "dog_ftn_roll5_rpo",
  "dog_ftn_roll5_play_action", "dog_ftn_roll5_pass_rush"
)

ngs_block <- c(
  "fav_ngs_roll5_cpoe", "fav_ngs_roll5_rush_yoe", "fav_ngs_roll5_rush_eff",
  "fav_ngs_roll5_ttt", "fav_ngs_roll5_aggressiveness",
  "dog_ngs_roll5_cpoe", "dog_ngs_roll5_rush_yoe", "dog_ngs_roll5_rush_eff",
  "dog_ngs_roll5_ttt", "dog_ngs_roll5_aggressiveness"
)

pfr_block <- c(
  "fav_pfr_roll5_pressure_rate", "fav_pfr_roll5_drop_rate", "fav_pfr_roll5_bad_throw_rate",
  "dog_pfr_roll5_pressure_rate", "dog_pfr_roll5_drop_rate", "dog_pfr_roll5_bad_throw_rate",
  "fav_pfr_roll5_def_pressure", "dog_pfr_roll5_def_pressure"
)

blocks <- list(
  FTN = ftn_block,
  NGS = ngs_block,
  PFR = pfr_block
)

# ============================================================
# TEST 1: Out-of-sample delta in log loss
# For each block: compare log loss of (base) vs (base + block)
# Using walk-forward on 2022+ data (where all blocks exist)
# ============================================================
cat("=== TEST 1: Out-of-sample log loss delta ===\n\n")

ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("classification")
ridge_grid <- grid_regular(penalty(range = c(-5, 1)), levels = 20)

# Restrict to 2022+ where all blocks are available
data_2022 <- model_data_v2 |> filter(season >= 2022)

run_block_test <- function(data, feature_vars, label, test_seasons = c(2024, 2025)) {
  preds_all <- list()

  for (test_s in test_seasons) {
    train <- data |> filter(season < test_s) |> select(any_of(feature_vars))
    test <- data |> filter(season == test_s) |> select(any_of(feature_vars))

    if (nrow(train) < 200) next

    inner_cv <- make_temporal_cv(train, min_train_season = min(train$season))

    rec <- recipe(fav_cover ~ ., data = train) |>
      update_role(game_id_norm, season, week, new_role = "id variable") |>
      step_zv(all_predictors()) |>
      step_impute_mean(all_numeric_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_normalize(all_numeric_predictors())

    wf <- workflow() |> add_model(ridge_spec) |> add_recipe(rec)

    set.seed(301)
    tune_res <- tune_grid(wf, resamples = inner_cv, grid = ridge_grid,
                          metrics = metric_set(mn_log_loss),
                          control = control_grid(save_pred = FALSE, verbose = FALSE))

    best <- select_best(tune_res, metric = "mn_log_loss")
    final_fit <- fit(finalize_workflow(wf, best), data = train)

    preds <- test |>
      select(game_id_norm, season, week, fav_cover) |>
      bind_cols(predict(final_fit, new_data = test, type = "prob"))

    preds_all[[as.character(test_s)]] <- preds
  }

  all_preds <- bind_rows(preds_all)
  ll <- log_loss(all_preds$fav_cover, all_preds$.pred_cover, "first")
  bs <- brier_score(all_preds$fav_cover, all_preds$.pred_cover, "first")
  acc <- mean((all_preds$.pred_cover > 0.5) == (all_preds$fav_cover == "cover"))

  list(label = label, log_loss = ll, brier = bs, accuracy = acc, preds = all_preds)
}

# Run base model
cat("Base model...\n")
base_result <- run_block_test(data_2022, base_vars, "base")
cat("  Log loss:", round(base_result$log_loss, 4), "\n")

# Run base + each block
block_results <- list(base = base_result)
for (block_name in names(blocks)) {
  cat(block_name, "block...\n")
  combined_vars <- c(base_vars, blocks[[block_name]])
  result <- run_block_test(data_2022, combined_vars, paste0("base+", block_name))
  delta <- base_result$log_loss - result$log_loss  # positive = block helped
  cat("  Log loss:", round(result$log_loss, 4), "| Delta:", round(delta, 4),
      if (delta > 0) "(IMPROVED)" else "(no improvement)", "\n")
  block_results[[block_name]] <- result
}

# All blocks combined
cat("All blocks combined...\n")
all_blocks_vars <- c(base_vars, unlist(blocks))
all_result <- run_block_test(data_2022, all_blocks_vars, "base+ALL")
delta_all <- base_result$log_loss - all_result$log_loss
cat("  Log loss:", round(all_result$log_loss, 4), "| Delta:", round(delta_all, 4),
    if (delta_all > 0) "(IMPROVED)" else "(no improvement)", "\n")
block_results[["ALL"]] <- all_result

# ============================================================
# TEST 2: Permutation null (is the observed lift unusual?)
# Shuffle each block's features, retrain, compare
# ============================================================
cat("\n=== TEST 2: Permutation null (B=20) ===\n")
cat("Shuffles block features across rows to destroy signal\n\n")

n_perms <- 10

for (block_name in names(blocks)) {
  cat(block_name, "block permutation test:\n")

  observed_delta <- base_result$log_loss -
    block_results[[block_name]]$log_loss

  perm_deltas <- numeric(n_perms)

  for (b in 1:n_perms) {
    # Shuffle the block features (break association with outcome)
    data_perm <- data_2022
    for (feat in blocks[[block_name]]) {
      if (feat %in% names(data_perm)) {
        data_perm[[feat]] <- sample(data_perm[[feat]])
      }
    }

    combined_vars <- c(base_vars, blocks[[block_name]])
    perm_result <- run_block_test(data_perm, combined_vars,
                                   paste0("perm_", b))
    perm_deltas[b] <- base_result$log_loss - perm_result$log_loss
  }

  p_value <- mean(perm_deltas >= observed_delta)
  cat("  Observed delta:", round(observed_delta, 5), "\n")
  cat("  Permutation mean:", round(mean(perm_deltas), 5),
      "sd:", round(sd(perm_deltas), 5), "\n")
  cat("  Permutation p-value:", round(p_value, 3),
      if (p_value < 0.10) "* SIGNIFICANT at 10%" else "", "\n\n")
}

# ============================================================
# SUMMARY TABLE
# ============================================================
cat("\n=== FEATURE BLOCK VALIDATION SUMMARY ===\n\n")

summary_table <- tibble(
  block = c("Base only", names(blocks), "ALL blocks"),
  log_loss = map_dbl(c("base", names(blocks), "ALL"),
                     ~ block_results[[.x]]$log_loss),
  brier = map_dbl(c("base", names(blocks), "ALL"),
                  ~ block_results[[.x]]$brier),
  accuracy = map_dbl(c("base", names(blocks), "ALL"),
                     ~ block_results[[.x]]$accuracy)
) |>
  mutate(
    delta_log_loss = log_loss[1] - log_loss,
    delta_brier = brier[1] - brier
  )

print(summary_table)

write_csv(summary_table, here("results", "feature_block_validation.csv"))
cat("\nDone!\n")
