# Advanced Calibration
# The Brier decomposition (script 20) showed calibration error is 18x discrimination.
# Current Platt scaling assumes linear logit correction — but miscalibration varies
# by spread bucket, QB status, and weather. We need non-parametric methods.
#
# Methods tested:
# 1. Platt scaling (current baseline — logistic on logit)
# 2. Isotonic regression (non-parametric, monotone)
# 3. Beta calibration (parametric, 3 params — more flexible than Platt)
# 4. Segment-aware calibration (different calibration per spread bucket)
# 5. Venn-Abers prediction (produces intervals, bet only when both bounds agree)
#
# All methods fitted inside training window, applied to test predictions.

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "14_research_protocol.R"))

# ============================================================
# 1. LOAD DATA AND RAW MODEL PREDICTIONS
# ============================================================

# Load v3 for features needed for segment-aware calibration
v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

# We need RAW (uncalibrated) predictions to test different calibration methods.
# Re-run walk-forward to get both raw and re-calibrate with each method.

cat("=== ADVANCED CALIBRATION ===\n")
cat("Rebuilding raw predictions with walk-forward...\n\n")

# Feature set (universal)
universal_vars <- c(
  "fav_cover", "game_id_norm", "season", "week",
  "spread_fav", "over_under_line", "elo_diff",
  "ypp_diff", "pass_eff_diff", "rush_eff_diff", "turnover_diff",
  "fav_roll5_ypp", "dog_roll5_pts", "fav_roll5_pts", "dog_roll5_ypp",
  "pts_play_diff", "third_down_diff", "sack_rate_diff",
  "off_epa_diff", "def_epa_diff", "success_rate_diff",
  "qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
  "third_down_epa_diff", "redzone_epa_diff",
  "pass_epa_mismatch", "rush_epa_mismatch",
  "explosive_diff", "neg_play_diff",
  "injury_diff", "fav_qb_out", "dog_qb_out",
  "div_game", "is_primetime", "is_dome",
  "rest_diff", "fav_at_home", "fav_short_week", "dog_short_week",
  "stadium_neutral", "weather_temperature", "weather_wind_mph",
  "momentum_diff", "to_momentum_diff",
  # New v3 engineered features
  "spread_squared", "elo_spread_residual", "efficiency_vs_market",
  "wind_pass_interaction", "pass_rush_mismatch", "total_epa_edge"
)

# Use existing available vars
available_vars <- intersect(universal_vars, names(v3))
cat("Using", length(available_vars), "features\n")

# Ridge spec
ridge_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |> set_mode("classification")
ridge_grid <- grid_regular(penalty(range = c(-5, 2)), levels = 40)

# ============================================================
# 2. WALK-FORWARD WITHOUT CALIBRATION (get raw predictions)
# ============================================================
cat("\nGenerating raw (uncalibrated) predictions...\n")

test_seasons <- c(2023, 2024, 2025)
raw_preds <- walk_forward_evaluate(
  v3, available_vars, ridge_spec, ridge_grid,
  min_train_season = 2012, test_seasons = test_seasons,
  calibrate = FALSE  # RAW predictions
)

cat("Raw predictions:", nrow(raw_preds), "games\n")

# Also get inner-fold validation predictions for calibration fitting
# Re-run but capture inner validation preds
cal_train_preds <- list()

for (test_s in test_seasons) {
  train_data <- v3 |>
    filter(season >= 2012, season < test_s) |>
    select(any_of(available_vars))

  inner_cv <- make_temporal_cv(train_data, min_train_season = 2012)

  recipe_fresh <- recipe(fav_cover ~ ., data = train_data) |>
    update_role(game_id_norm, season, week, new_role = "id variable") |>
    step_zv(all_predictors()) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

  wf <- workflow() |> add_model(ridge_spec) |> add_recipe(recipe_fresh)

  set.seed(301)
  tune_result <- tune_grid(
    wf, resamples = inner_cv, grid = ridge_grid,
    metrics = metric_set(mn_log_loss),
    control = control_grid(save_pred = TRUE, verbose = FALSE)
  )

  best <- select_best(tune_result, metric = "mn_log_loss")
  inner_preds <- collect_predictions(tune_result) |>
    semi_join(best, by = intersect(names(best), names(collect_predictions(tune_result))))

  cal_train_preds[[as.character(test_s)]] <- inner_preds |>
    mutate(for_test_season = test_s)
}

# ============================================================
# 3. CALIBRATION METHOD 1: PLATT SCALING (baseline)
# ============================================================
cat("\n=== METHOD 1: Platt Scaling ===\n")

platt_preds <- raw_preds
for (test_s in test_seasons) {
  inner <- cal_train_preds[[as.character(test_s)]]
  cal_fit <- fit_platt(inner)

  mask <- platt_preds$test_season == test_s
  logits <- log(platt_preds$.pred_cover[mask] / (1 - platt_preds$.pred_cover[mask]))
  cal_probs <- predict(cal_fit, newdata = data.frame(.pred_cover = platt_preds$.pred_cover[mask]),
                        type = "response")
  platt_preds$.pred_calibrated[mask] <- pmin(pmax(cal_probs, 0.01), 0.99)
}

platt_brier <- brier_score(platt_preds$fav_cover, platt_preds$.pred_calibrated, "first")
platt_ll <- log_loss(platt_preds$fav_cover, platt_preds$.pred_calibrated, "first")
platt_acc <- mean((platt_preds$.pred_calibrated > 0.5) == (platt_preds$fav_cover == "cover"))
cat(sprintf("  Brier: %.5f  Log loss: %.5f  Accuracy: %.1f%%\n", platt_brier, platt_ll, platt_acc * 100))

# ============================================================
# 4. CALIBRATION METHOD 2: ISOTONIC REGRESSION
# ============================================================
cat("\n=== METHOD 2: Isotonic Regression ===\n")

iso_preds <- raw_preds
iso_preds$.pred_calibrated <- NA_real_

for (test_s in test_seasons) {
  inner <- cal_train_preds[[as.character(test_s)]]

  # Fit isotonic regression on inner validation predictions
  truth_num <- as.integer(inner$fav_cover == "cover")
  # isoreg requires sorted x values
  ord <- order(inner$.pred_cover)
  iso_fit <- isoreg(inner$.pred_cover[ord], truth_num[ord])

  # Create step function for applying to new data
  knots <- iso_fit$iKnots
  if (length(knots) > 0) {
    iso_stepfun <- stepfun(iso_fit$x[knots], c(iso_fit$yf[1], iso_fit$yf[knots]))
  } else {
    # Fallback: constant function at observed mean
    iso_stepfun <- function(x) rep(mean(truth_num), length(x))
  }

  mask <- iso_preds$test_season == test_s
  cal_probs <- iso_stepfun(iso_preds$.pred_cover[mask])
  iso_preds$.pred_calibrated[mask] <- pmin(pmax(cal_probs, 0.01), 0.99)
}

iso_brier <- brier_score(iso_preds$fav_cover, iso_preds$.pred_calibrated, "first")
iso_ll <- log_loss(iso_preds$fav_cover, iso_preds$.pred_calibrated, "first")
iso_acc <- mean((iso_preds$.pred_calibrated > 0.5) == (iso_preds$fav_cover == "cover"))
cat(sprintf("  Brier: %.5f  Log loss: %.5f  Accuracy: %.1f%%\n", iso_brier, iso_ll, iso_acc * 100))

# ============================================================
# 5. CALIBRATION METHOD 3: BETA CALIBRATION
# ============================================================
cat("\n=== METHOD 3: Beta Calibration ===\n")

# Beta calibration: logit(q) = a * log(p/(1-p)) + b * log(p) + c
# More flexible than Platt (3 params vs 2)
beta_preds <- raw_preds
beta_preds$.pred_calibrated <- NA_real_

for (test_s in test_seasons) {
  inner <- cal_train_preds[[as.character(test_s)]]
  truth_num <- as.integer(inner$fav_cover == "cover")
  p <- pmin(pmax(inner$.pred_cover, 0.001), 0.999)

  # Beta calibration features
  logit_p <- log(p / (1 - p))
  log_p <- log(p)

  beta_fit <- glm(truth_num ~ logit_p + log_p, family = binomial())

  mask <- beta_preds$test_season == test_s
  p_test <- pmin(pmax(beta_preds$.pred_cover[mask], 0.001), 0.999)
  new_data <- data.frame(logit_p = log(p_test / (1 - p_test)), log_p = log(p_test))
  cal_probs <- predict(beta_fit, newdata = new_data, type = "response")
  beta_preds$.pred_calibrated[mask] <- pmin(pmax(cal_probs, 0.01), 0.99)
}

beta_brier <- brier_score(beta_preds$fav_cover, beta_preds$.pred_calibrated, "first")
beta_ll <- log_loss(beta_preds$fav_cover, beta_preds$.pred_calibrated, "first")
beta_acc <- mean((beta_preds$.pred_calibrated > 0.5) == (beta_preds$fav_cover == "cover"))
cat(sprintf("  Brier: %.5f  Log loss: %.5f  Accuracy: %.1f%%\n", beta_brier, beta_ll, beta_acc * 100))

# ============================================================
# 6. CALIBRATION METHOD 4: SEGMENT-AWARE CALIBRATION
# ============================================================
cat("\n=== METHOD 4: Segment-Aware (spread bucket + isotonic) ===\n")

# Different calibration per spread bucket — addresses the non-uniform miscalibration
seg_preds <- raw_preds
seg_preds$.pred_calibrated <- NA_real_

# Add spread info
seg_preds <- seg_preds |>
  left_join(v3 |> select(game_id_norm, spread_fav), by = "game_id_norm") |>
  mutate(
    abs_spread = abs(spread_fav),
    spread_group = case_when(
      abs_spread <= 3   ~ "tight",
      abs_spread <= 7   ~ "medium",
      TRUE              ~ "large"
    )
  )

for (test_s in test_seasons) {
  inner <- cal_train_preds[[as.character(test_s)]] |>
    left_join(v3 |> select(game_id_norm, spread_fav), by = "game_id_norm") |>
    mutate(
      abs_spread = abs(spread_fav),
      spread_group = case_when(
        abs_spread <= 3   ~ "tight",
        abs_spread <= 7   ~ "medium",
        TRUE              ~ "large"
      )
    )

  for (grp in unique(inner$spread_group)) {
    inner_grp <- inner |> filter(spread_group == grp)
    if (nrow(inner_grp) < 30) next

    truth_num <- as.integer(inner_grp$fav_cover == "cover")
    ord <- order(inner_grp$.pred_cover)
    iso_fit <- isoreg(inner_grp$.pred_cover[ord], truth_num[ord])
    knots <- iso_fit$iKnots
    if (length(knots) > 0) {
      iso_fn <- stepfun(iso_fit$x[knots], c(iso_fit$yf[1], iso_fit$yf[knots]))
    } else {
      m <- mean(truth_num)
      iso_fn <- function(x) rep(m, length(x))
    }

    mask <- seg_preds$test_season == test_s & seg_preds$spread_group == grp
    if (sum(mask) == 0) next
    seg_preds$.pred_calibrated[mask] <- pmin(pmax(
      iso_fn(seg_preds$.pred_cover[mask]), 0.01), 0.99)
  }
}

# Fill any remaining NAs with Platt fallback
na_mask <- is.na(seg_preds$.pred_calibrated)
if (any(na_mask)) {
  seg_preds$.pred_calibrated[na_mask] <- platt_preds$.pred_calibrated[na_mask]
}

seg_brier <- brier_score(seg_preds$fav_cover, seg_preds$.pred_calibrated, "first")
seg_ll <- log_loss(seg_preds$fav_cover, seg_preds$.pred_calibrated, "first")
seg_acc <- mean((seg_preds$.pred_calibrated > 0.5) == (seg_preds$fav_cover == "cover"))
cat(sprintf("  Brier: %.5f  Log loss: %.5f  Accuracy: %.1f%%\n", seg_brier, seg_ll, seg_acc * 100))

# ============================================================
# 7. CALIBRATION METHOD 5: VENN-ABERS PREDICTION
# ============================================================
cat("\n=== METHOD 5: Venn-Abers Intervals ===\n")

# Venn-Abers: for each test point, fit isotonic regression twice:
# once assuming it's class 0, once assuming it's class 1.
# Produces interval [p0, p1]. Bet only when both endpoints agree.

va_preds <- raw_preds
va_preds$va_lower <- NA_real_
va_preds$va_upper <- NA_real_

for (test_s in test_seasons) {
  inner <- cal_train_preds[[as.character(test_s)]]
  truth_num <- as.integer(inner$fav_cover == "cover")
  inner_scores <- inner$.pred_cover

  mask <- which(va_preds$test_season == test_s)

  # Venn-Abers is O(n^2) per test point — subsample if > 100
  if (length(mask) > 100) {
    set.seed(301 + test_s)
    mask <- sample(mask, 100)
    cat(sprintf("  Season %d: subsampled to 100 games for Venn-Abers\n", test_s))
  }

  for (i in seq_along(mask)) {
    idx <- mask[i]
    test_score <- va_preds$.pred_cover[idx]

    # Augment training set with test point as class 0
    scores_0 <- c(inner_scores, test_score)
    labels_0 <- c(truth_num, 0)
    ord_0 <- order(scores_0)
    iso_0 <- isoreg(scores_0[ord_0], labels_0[ord_0])
    k0 <- iso_0$iKnots
    if (length(k0) > 0) {
      fn_0 <- stepfun(iso_0$x[k0], c(iso_0$yf[1], iso_0$yf[k0]))
    } else {
      fn_0 <- function(x) rep(mean(labels_0), length(x))
    }
    p0 <- fn_0(test_score)

    # Augment training set with test point as class 1
    scores_1 <- c(inner_scores, test_score)
    labels_1 <- c(truth_num, 1)
    ord_1 <- order(scores_1)
    iso_1 <- isoreg(scores_1[ord_1], labels_1[ord_1])
    k1 <- iso_1$iKnots
    if (length(k1) > 0) {
      fn_1 <- stepfun(iso_1$x[k1], c(iso_1$yf[1], iso_1$yf[k1]))
    } else {
      fn_1 <- function(x) rep(mean(labels_1), length(x))
    }
    p1 <- fn_1(test_score)

    va_preds$va_lower[idx] <- min(p0, p1)
    va_preds$va_upper[idx] <- max(p0, p1)
  }

  cat(sprintf("  Season %d: %d games processed\n", test_s, length(mask)))
}

# Point estimate = midpoint of interval
va_preds$.pred_calibrated <- (va_preds$va_lower + va_preds$va_upper) / 2
va_preds$va_width <- va_preds$va_upper - va_preds$va_lower

va_brier <- brier_score(va_preds$fav_cover, va_preds$.pred_calibrated, "first")
va_ll <- log_loss(va_preds$fav_cover, va_preds$.pred_calibrated, "first")
va_acc <- mean((va_preds$.pred_calibrated > 0.5) == (va_preds$fav_cover == "cover"))
cat(sprintf("  Brier: %.5f  Log loss: %.5f  Accuracy: %.1f%%\n", va_brier, va_ll, va_acc * 100))

# Venn-Abers betting: only bet when BOTH interval endpoints agree on side
va_agree <- va_preds |>
  mutate(
    both_cover = va_lower > 0.5 & va_upper > 0.5,
    both_nocover = va_lower < 0.5 & va_upper < 0.5,
    confident = both_cover | both_nocover,
    bet_cover = both_cover,
    correct = (bet_cover == (fav_cover == "cover"))
  )

n_confident <- sum(va_agree$confident)
if (n_confident > 0) {
  conf_acc <- mean(va_agree$correct[va_agree$confident])
  cat(sprintf("\n  Confident bets (both bounds agree): %d/%d (%.1f%%)\n",
              n_confident, nrow(va_agree), n_confident / nrow(va_agree) * 100))
  cat(sprintf("  Confident bet accuracy: %.1f%%\n", conf_acc * 100))
  cat(sprintf("  Breakeven needed: 52.4%%\n"))
} else {
  cat("  No confident bets (model too uncertain)\n")
}

# ============================================================
# 8. COMPARISON TABLE
# ============================================================
cat("\n=== CALIBRATION COMPARISON ===\n")

comparison <- tibble(
  method = c("Uncalibrated", "Platt", "Isotonic", "Beta", "Segment-Aware", "Venn-Abers"),
  brier = c(
    brier_score(raw_preds$fav_cover, raw_preds$.pred_cover, "first"),
    platt_brier, iso_brier, beta_brier, seg_brier, va_brier
  ),
  log_loss = c(
    log_loss(raw_preds$fav_cover, raw_preds$.pred_cover, "first"),
    platt_ll, iso_ll, beta_ll, seg_ll, va_ll
  ),
  accuracy = c(
    mean((raw_preds$.pred_cover > 0.5) == (raw_preds$fav_cover == "cover")),
    platt_acc, iso_acc, beta_acc, seg_acc, va_acc
  )
) |>
  mutate(
    brier_vs_coinflip = brier - 0.25,
    ll_vs_coinflip = log_loss - 0.6931,
    rank = rank(brier)
  )

cat("\n")
print(comparison |> arrange(brier))

write_csv(comparison, here("results", "calibration_comparison.csv"))

# ============================================================
# 9. BETTING ANALYSIS WITH BEST CALIBRATION METHOD
# ============================================================
cat("\n=== BETTING WITH BEST CALIBRATION ===\n")

# Find best method
best_method <- comparison |> slice_min(brier) |> pull(method)
cat("Best calibration method:", best_method, "\n\n")

# Use the best calibrated predictions
if (best_method == "Isotonic") {
  best_preds <- iso_preds
} else if (best_method == "Beta") {
  best_preds <- beta_preds
} else if (best_method == "Segment-Aware") {
  best_preds <- seg_preds
} else if (best_method == "Venn-Abers") {
  best_preds <- va_preds
} else {
  best_preds <- platt_preds
}

# EV-based betting with calibrated probabilities
best_preds <- best_preds |>
  mutate(
    cal_p = .pred_calibrated,
    ev_cover = compute_ev(cal_p),
    ev_nocover = compute_ev(1 - cal_p),
    best_ev = pmax(ev_cover, ev_nocover),
    bet_side = if_else(ev_cover > ev_nocover, "cover", "no_cover"),
    truth_str = if_else(fav_cover == "cover", "cover", "no_cover"),
    correct = bet_side == truth_str
  )

# EV threshold analysis
cat("EV-based betting with", best_method, "calibration:\n")
for (thresh in c(0, 1, 2, 3, 5)) {
  bets <- best_preds |> filter(best_ev > thresh)
  if (nrow(bets) == 0) next
  win_rate <- mean(bets$correct)
  n_bets <- nrow(bets)
  profit <- sum(if_else(bets$correct, 100, -110))
  roi <- profit / (n_bets * 110)

  # Wilson CI
  se <- sqrt(win_rate * (1 - win_rate) / n_bets)
  ci_lower <- win_rate - 1.96 * se
  ci_upper <- win_rate + 1.96 * se

  cat(sprintf("  EV > $%d: %4d bets (%4.1f%%) | win: %.1f%% [%.1f%%-%.1f%%] | profit: $%6.0f | ROI: %5.1f%%\n",
              thresh, n_bets, n_bets / nrow(best_preds) * 100,
              win_rate * 100, ci_lower * 100, ci_upper * 100,
              profit, roi * 100))
}

# Save best predictions
best_preds_save <- best_preds |>
  select(game_id_norm, season, week, fav_cover, .pred_cover,
         .pred_calibrated, cal_p, ev_cover, ev_nocover, best_ev, bet_side, correct,
         any_of(c("va_lower", "va_upper", "va_width")))
write_rds(best_preds_save, here("objects", "calibrated_predictions.rds"))

cat("\nDone! Best calibration method:", best_method, "\n")
cat("Predictions saved to objects/calibrated_predictions.rds\n")
