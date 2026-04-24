# Adversarial Temporal Analysis
# Determines if the 2024-2025 model collapse is distributional shift or overfitting
# Tests: adversarial validation, feature drift, coefficient stability

library(tidyverse)
library(tidymodels)
library(here)
library(ranger)

source(here("scripts", "14_research_protocol.R"))

v3 <- read_rds(here("objects", "modeling_data_v3.rds"))

# ============================================================
# 1. ADVERSARIAL VALIDATION
# ============================================================
cat("=== ADVERSARIAL VALIDATION ===\n")
cat("Can a classifier distinguish train (2012-2022) from test (2023-2025)?\n\n")

# Create binary label: is this game from test era?
adv_data <- v3 |>
  mutate(is_test = factor(if_else(season >= 2023, "test", "train"))) |>
  select(is_test, where(is.numeric), -season, -week) |>
  drop_na()

cat("Adversarial data:", nrow(adv_data), "rows,", ncol(adv_data) - 1, "features\n")

# Train random forest to distinguish eras
set.seed(301)
adv_split <- initial_split(adv_data, prop = 0.7, strata = is_test)
adv_train <- training(adv_split)
adv_test <- testing(adv_split)

adv_rf <- ranger(
  is_test ~ ., data = adv_train,
  num.trees = 500, probability = TRUE, importance = "impurity"
)

adv_pred <- predict(adv_rf, adv_test)$predictions[, "test"]
adv_auc <- roc_auc_vec(adv_test$is_test, adv_pred, event_level = "second")

cat(sprintf("Adversarial AUC: %.3f\n", adv_auc))
if (adv_auc > 0.55) {
  cat("  WARNING: AUC > 0.55 — significant distributional shift detected.\n")
  cat("  The model can distinguish train from test eras using features alone.\n")
  cat("  This means feature distributions changed between eras.\n")
} else {
  cat("  OK: AUC < 0.55 — train and test distributions are similar.\n")
  cat("  The 2024-2025 collapse is likely overfitting, not shift.\n")
}

# Top features for distinguishing eras
imp <- tibble(
  feature = names(adv_rf$variable.importance),
  importance = adv_rf$variable.importance
) |> arrange(desc(importance))

cat("\nTop 15 features distinguishing train from test:\n")
print(imp |> head(15))

# ============================================================
# 2. FEATURE DRIFT (Kolmogorov-Smirnov)
# ============================================================
cat("\n=== FEATURE DRIFT (KS test) ===\n")

train_seasons <- 2012:2022
numeric_cols <- v3 |> select(where(is.numeric), -season, -week) |> names()

drift_results <- map(numeric_cols, function(col) {
  train_vals <- v3 |> filter(season %in% train_seasons) |> pull(!!sym(col)) |> na.omit()

  ks_by_season <- map_dfr(2023:2025, function(test_s) {
    test_vals <- v3 |> filter(season == test_s) |> pull(!!sym(col)) |> na.omit()
    if (length(test_vals) < 10 || length(train_vals) < 10) {
      return(tibble(feature = col, test_season = test_s, ks_stat = NA, ks_p = NA))
    }
    ks <- ks.test(train_vals, test_vals)
    tibble(feature = col, test_season = test_s, ks_stat = ks$statistic, ks_p = ks$p.value)
  })
}) |> bind_rows()

# Features with significant drift (p < 0.05 in any test season)
significant_drift <- drift_results |>
  filter(ks_p < 0.05) |>
  group_by(feature) |>
  summarize(max_ks = max(ks_stat, na.rm = TRUE), n_seasons_drifted = n(), .groups = "drop") |>
  arrange(desc(max_ks))

cat("\nFeatures with significant drift (KS p < 0.05):\n")
print(significant_drift |> head(20))
cat(sprintf("\n%d of %d features show significant drift\n",
            nrow(significant_drift), length(numeric_cols)))

# ============================================================
# 3. COEFFICIENT STABILITY
# ============================================================
cat("\n=== COEFFICIENT STABILITY ===\n")

# Train ridge on expanding windows, track coefficient changes
universal_vars <- c(
  "spread_fav", "over_under_line", "elo_diff",
  "ypp_diff", "pass_eff_diff", "rush_eff_diff", "turnover_diff",
  "off_epa_diff", "def_epa_diff", "success_rate_diff",
  "qb_epa_diff", "qb_cpoe_diff", "injury_diff",
  "rest_diff", "weather_temperature", "weather_wind_mph",
  "momentum_diff", "pass_epa_mismatch", "rush_epa_mismatch",
  "explosive_diff", "spread_squared", "elo_spread_residual"
)
available_uni <- intersect(universal_vars, names(v3))

coef_paths <- list()
for (end_season in 2018:2024) {
  train <- v3 |>
    filter(season >= 2012, season <= end_season) |>
    select(fav_cover, all_of(available_uni)) |>
    drop_na()

  recipe_obj <- recipe(fav_cover ~ ., data = train) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors()) |>
    prep()

  processed <- bake(recipe_obj, train)
  x <- as.matrix(processed |> select(-fav_cover))
  y <- as.integer(processed$fav_cover == "cover")

  fit <- glmnet::glmnet(x, y, family = "binomial", alpha = 0, lambda = 0.01)
  coefs <- as.matrix(coef(fit))[-1, 1]  # drop intercept

  coef_paths[[as.character(end_season)]] <- tibble(
    end_season = end_season,
    feature = names(coefs),
    coefficient = coefs
  )
}

coef_df <- bind_rows(coef_paths)

# Find features with unstable signs
sign_changes <- coef_df |>
  group_by(feature) |>
  summarize(
    sign_changes = sum(diff(sign(coefficient)) != 0),
    mean_coef = mean(coefficient),
    sd_coef = sd(coefficient),
    cv = sd_coef / (abs(mean_coef) + 1e-6),
    .groups = "drop"
  ) |>
  arrange(desc(sign_changes))

cat("Features with unstable coefficient signs:\n")
print(sign_changes |> filter(sign_changes > 0))

cat("\nMost stable features (never change sign):\n")
print(sign_changes |> filter(sign_changes == 0, abs(mean_coef) > 0.01) |>
        arrange(desc(abs(mean_coef))))

# ============================================================
# 4. SAVE
# ============================================================
write_csv(drift_results, here("results", "feature_drift.csv"))
write_csv(significant_drift, here("results", "significant_drift_features.csv"))
write_csv(coef_df, here("results", "coefficient_paths.csv"))

cat("\nDone!\n")
