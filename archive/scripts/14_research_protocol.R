# Research Protocol: Defines the validation framework for all subsequent scripts
# All tuning, calibration, and evaluation uses nested walk-forward validation
# No random CV. No in-sample threshold optimization. No post-hoc slicing.

library(tidyverse)
library(tidymodels)
library(here)
library(probably)
library(slider)

source(here("scripts", "00_project_setup.R"))

# ============================================================
# 1. TEMPORAL CROSS-VALIDATION (replaces random v-fold)
# ============================================================

# Create temporal CV splits: train on seasons [start, end], validate on end+1
make_temporal_cv <- function(data, min_train_season = 2018, max_val_season = NULL) {
  seasons <- sort(unique(data$season))
  if (is.null(max_val_season)) max_val_season <- max(seasons)

  val_seasons <- seasons[seasons > min_train_season & seasons <= max_val_season]

  splits <- map(val_seasons, function(val_s) {
    train_idx <- which(data$season >= min_train_season & data$season < val_s)
    val_idx <- which(data$season == val_s)
    make_splits(list(analysis = train_idx, assessment = val_idx), data)
  })

  manual_rset(splits, paste0("season_", val_seasons))
}

# ============================================================
# 2. CALIBRATION (Platt scaling fitted inside training window)
# ============================================================

# Fit Platt scaling (logistic recalibration) on validation predictions
fit_platt <- function(val_preds) {
  # val_preds must have .pred_cover and fav_cover columns
  cal_data <- val_preds |>
    mutate(outcome_num = as.integer(fav_cover == "cover"))

  # Logistic regression: outcome ~ logit(predicted_prob)
  cal_fit <- glm(
    outcome_num ~ log(.pred_cover / (1 - .pred_cover)),
    data = cal_data, family = binomial()
  )
  cal_fit
}

# Apply Platt scaling to new predictions
apply_platt <- function(new_preds, cal_fit) {
  logits <- log(new_preds$.pred_cover / (1 - new_preds$.pred_cover))
  cal_probs <- predict(cal_fit, newdata = data.frame(`.pred_cover` = new_preds$.pred_cover),
                       type = "response")
  # Handle edge cases
  cal_probs <- pmin(pmax(cal_probs, 0.01), 0.99)
  new_preds |> mutate(.pred_cover_calibrated = cal_probs)
}

# ============================================================
# 3. PROPER SCORING RULES
# ============================================================

# Brier score (lower is better, 0.25 = coin flip baseline)
brier_score <- function(truth_factor, pred_prob, event_level = "first") {
  truth_num <- if (event_level == "first") {
    as.integer(truth_factor == levels(truth_factor)[1])
  } else {
    as.integer(truth_factor == levels(truth_factor)[2])
  }
  mean((pred_prob - truth_num)^2)
}

# Log loss (lower is better)
log_loss <- function(truth_factor, pred_prob, event_level = "first") {
  truth_num <- if (event_level == "first") {
    as.integer(truth_factor == levels(truth_factor)[1])
  } else {
    as.integer(truth_factor == levels(truth_factor)[2])
  }
  pred_prob <- pmin(pmax(pred_prob, 1e-15), 1 - 1e-15)
  -mean(truth_num * log(pred_prob) + (1 - truth_num) * log(1 - pred_prob))
}

# ============================================================
# 4. EXPECTED VALUE CALCULATION
# ============================================================

# At standard -110 odds: risk $110 to win $100
# EV = p * 100 - (1 - p) * 110
# Breakeven: p = 110/210 = 0.5238

compute_ev <- function(calibrated_p, risk = 110, payout = 100) {
  calibrated_p * payout - (1 - calibrated_p) * risk
}

# EV with uncertainty: lower bound using normal approximation
# n = number of similar predictions used to estimate p
compute_ev_lower_bound <- function(calibrated_p, n_similar, risk = 110, payout = 100,
                                    confidence = 0.90) {
  se <- sqrt(calibrated_p * (1 - calibrated_p) / pmax(n_similar, 1))
  z <- qnorm((1 + confidence) / 2)
  p_lower <- calibrated_p - z * se
  compute_ev(p_lower, risk, payout)
}

# ============================================================
# 5. WALK-FORWARD EVALUATION FRAMEWORK
# ============================================================

# Full walk-forward pipeline for one model specification
walk_forward_evaluate <- function(data, feature_vars, spec, grid,
                                   min_train_season = 2018,
                                   test_seasons = NULL,
                                   calibrate = TRUE) {
  if (is.null(test_seasons)) {
    test_seasons <- sort(unique(data$season[data$season >= 2023]))
  }

  results <- map(test_seasons, function(test_s) {
    cat("  Evaluating season", test_s, "...\n")

    # --- Inner loop: tune on temporal CV within training window ---
    train_data <- data |>
      filter(season >= min_train_season, season < test_s) |>
      select(any_of(feature_vars))

    test_data <- data |>
      filter(season == test_s) |>
      select(any_of(feature_vars))

    if (nrow(train_data) < 200) {
      cat("    Skipping - only", nrow(train_data), "train rows\n")
      return(NULL)
    }

    # Temporal CV for tuning (leave last season as inner validation)
    inner_cv <- make_temporal_cv(train_data, min_train_season = min_train_season)

    recipe_fresh <- recipe(fav_cover ~ ., data = train_data) |>
      update_role(game_id_norm, season, week, new_role = "id variable") |>
      step_zv(all_predictors()) |>
      step_impute_mean(all_numeric_predictors()) |>
      step_dummy(all_nominal_predictors()) |>
      step_normalize(all_numeric_predictors())

    wf <- workflow() |> add_model(spec) |> add_recipe(recipe_fresh)

    set.seed(301)
    tune_result <- tune_grid(
      wf, resamples = inner_cv, grid = grid,
      metrics = metric_set(roc_auc, mn_log_loss),
      control = control_grid(save_pred = TRUE, verbose = FALSE)
    )

    best <- select_best(tune_result, metric = "mn_log_loss")  # Use log loss, not AUC

    # --- Calibration on inner validation predictions ---
    if (calibrate) {
      inner_preds <- collect_predictions(tune_result) |>
        semi_join(best, by = intersect(names(best), names(collect_predictions(tune_result))))
      cal_fit <- fit_platt(inner_preds |> rename(fav_cover = fav_cover))
    }

    # --- Fit final model on full training window ---
    final_wf <- finalize_workflow(wf, best)
    final_fit <- fit(final_wf, data = train_data)

    # --- Predict test season ---
    test_preds <- test_data |>
      select(game_id_norm, season, week, fav_cover) |>
      bind_cols(predict(final_fit, new_data = test_data, type = "prob")) |>
      bind_cols(predict(final_fit, new_data = test_data, type = "class"))

    # --- Apply calibration ---
    if (calibrate) {
      test_preds <- apply_platt(test_preds, cal_fit)
    } else {
      test_preds <- test_preds |> mutate(.pred_cover_calibrated = .pred_cover)
    }

    # --- Compute metrics ---
    test_preds |> mutate(
      test_season = test_s,
      train_start = min_train_season,
      train_end = test_s - 1,
      train_n = nrow(train_data)
    )
  }) |>
    compact() |>
    bind_rows()

  results
}

# ============================================================
# 6. REPORTING FUNCTIONS
# ============================================================

compute_season_metrics <- function(preds, prob_col = ".pred_cover_calibrated") {
  preds |>
    group_by(test_season) |>
    summarize(
      n = n(),
      brier = brier_score(fav_cover, .data[[prob_col]], event_level = "first"),
      log_loss = log_loss(fav_cover, .data[[prob_col]], event_level = "first"),
      roc_auc = tryCatch(
        roc_auc_vec(fav_cover, .data[[prob_col]], event_level = "first"),
        error = function(e) NA_real_
      ),
      accuracy = mean((.data[[prob_col]] > 0.5) == (fav_cover == "cover")),
      # Betting metrics with calibrated EV threshold
      n_positive_ev = sum(abs(compute_ev(.data[[prob_col]])) > 0 &
                            compute_ev(.data[[prob_col]]) > 0),
      ev_bets_accuracy = {
        ev_bets <- .data[[prob_col]][compute_ev(.data[[prob_col]]) > 0]
        truth_ev <- (fav_cover == "cover")[compute_ev(.data[[prob_col]]) > 0]
        if (length(ev_bets) > 0) mean((ev_bets > 0.5) == truth_ev) else NA_real_
      },
      .groups = "drop"
    )
}

# Calibration curve data
calibration_curve_data <- function(preds, prob_col = ".pred_cover_calibrated", n_bins = 10) {
  preds |>
    mutate(
      prob = .data[[prob_col]],
      outcome = as.integer(fav_cover == "cover"),
      bin = cut(prob, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
    ) |>
    group_by(bin) |>
    summarize(
      mean_predicted = mean(prob),
      mean_observed = mean(outcome),
      n = n(),
      se = sqrt(mean_observed * (1 - mean_observed) / pmax(n, 1)),
      .groups = "drop"
    )
}

cat("Research protocol loaded.\n")
cat("Available functions:\n")
cat("  make_temporal_cv()         - Temporal CV splits (no random folds)\n")
cat("  fit_platt() / apply_platt() - Platt scaling calibration\n")
cat("  brier_score() / log_loss() - Proper scoring rules\n")
cat("  compute_ev()               - Expected value at -110 odds\n")
cat("  compute_ev_lower_bound()   - EV with uncertainty bounds\n")
cat("  walk_forward_evaluate()    - Full nested walk-forward pipeline\n")
cat("  compute_season_metrics()   - Season-level reporting\n")
cat("  calibration_curve_data()   - Calibration curve for plotting\n")
