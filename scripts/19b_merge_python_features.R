# Merge Python feature store into R modeling framework
# Creates modeling_data_v3.rds from v2 + new Python features
# Coverage flags replace zero-imputation: NAs instead of misleading zeros

library(tidyverse)
library(here)
library(arrow)

source(here("scripts", "00_project_setup.R"))

# ============================================================
# 1. LOAD EXISTING R DATA
# ============================================================
cat("Loading modeling_data_v2.rds...\n")
v2 <- read_rds(here("objects", "modeling_data_v2.rds"))
cat("  v2:", nrow(v2), "rows x", ncol(v2), "columns\n")

# ============================================================
# 2. LOAD PYTHON FEATURE STORE
# ============================================================
feature_store_path <- here("data", "processed", "modeling", "feature_store_v1.parquet")

if (!file.exists(feature_store_path)) {
  cat("Python feature store not found. Run the Python pipeline first:\n")
  cat("  PYTHONPATH=. python3 scripts/run_pipeline.py\n")
  cat("Proceeding with v2 data only...\n")

  # Still create v3 with coverage flags added to v2
  v3 <- v2 |>
    mutate(
      has_pfr = season >= 2018,
      has_ngs = season >= 2016,
      has_ftn = season >= 2022,
      # De-vig implied probabilities from moneylines if available
      spread_squared = spread_fav^2,
      abs_spread = abs(spread_fav)
    )

} else {
  cat("Loading Python feature store...\n")
  py_store <- read_parquet(feature_store_path) |> as_tibble()
  cat("  Python store:", nrow(py_store), "rows x", ncol(py_store), "columns\n")

  # Identify new columns in Python store not in v2
  v2_cols <- names(v2)
  py_cols <- names(py_store)
  new_cols <- setdiff(py_cols, v2_cols)
  shared_cols <- intersect(py_cols, v2_cols)

  cat("  Shared columns:", length(shared_cols), "\n")
  cat("  New columns from Python:", length(new_cols), "\n")
  if (length(new_cols) > 0) {
    cat("  New columns:", paste(head(new_cols, 20), collapse=", "), "\n")
  }

  # Join new columns onto v2 by game_id_norm
  if ("game_id_norm" %in% shared_cols & length(new_cols) > 0) {
    py_new <- py_store |> select(game_id_norm, any_of(new_cols))
    v3 <- v2 |> left_join(py_new, by = "game_id_norm")
  } else {
    v3 <- v2
  }

  # Ensure coverage flags exist
  if (!"has_pfr" %in% names(v3)) {
    v3 <- v3 |> mutate(has_pfr = season >= 2018)
  }
  if (!"has_ngs" %in% names(v3)) {
    v3 <- v3 |> mutate(has_ngs = season >= 2016)
  }
  if (!"has_ftn" %in% names(v3)) {
    v3 <- v3 |> mutate(has_ftn = season >= 2022)
  }
}

# ============================================================
# 3. ADD ENGINEERED FEATURES (new for v3)
# ============================================================
cat("\nAdding engineered features...\n")

v3 <- v3 |>
  mutate(
    # Non-linear spread effect
    spread_squared = spread_fav^2,
    abs_spread = abs(spread_fav),

    # Spread buckets (key numbers)
    spread_bucket = case_when(
      abs_spread <= 3   ~ "tiny_1_3",
      abs_spread <= 6.5 ~ "small_3.5_6.5",
      abs_spread <= 9.5 ~ "medium_7_9.5",
      abs_spread <= 13.5 ~ "large_10_13.5",
      TRUE               ~ "huge_14_plus"
    ),

    # Elo-implied spread residual (market disagreement)
    # Elo implies ~2.7 points per 100 elo diff
    elo_implied_spread = elo_diff * 2.7 / 100,
    elo_spread_residual = elo_implied_spread - spread_fav,

    # EPA vs market gap
    efficiency_vs_market = off_epa_diff - (spread_fav / 14),  # rough scaling

    # QB uncertainty interaction
    qb_uncertainty_x_spread = as.numeric(fav_qb_out == TRUE | dog_qb_out == TRUE) * abs_spread,

    # Wind x passing (only meaningful outdoors)
    wind_pass_interaction = pmax(0, weather_wind_mph - 12, na.rm = TRUE) * pass_eff_diff,

    # Rest interactions
    rest_advantage = pmax(rest_diff, 0),
    fav_rested_dog_short = as.numeric(fav_short_week == "no" & dog_short_week == "yes"),

    # Mismatch composites
    pass_rush_mismatch = pass_epa_mismatch - rush_epa_mismatch,
    total_epa_edge = off_epa_diff - def_epa_diff,

    # Season timing
    season_phase = case_when(
      week <= 4  ~ "early",
      week <= 12 ~ "mid",
      TRUE       ~ "late"
    )
  )

# ============================================================
# 4. VALIDATE
# ============================================================
cat("\n=== VALIDATION ===\n")
cat("v3 dimensions:", nrow(v3), "x", ncol(v3), "\n")
cat("Seasons:", min(v3$season), "-", max(v3$season), "\n")

# Check no NAs in critical columns
critical <- c("spread_fav", "elo_diff", "fav_cover", "season", "week")
for (col in critical) {
  na_count <- sum(is.na(v3[[col]]))
  cat("  ", col, "NAs:", na_count, "\n")
}

# Games per season
cat("\nGames per season:\n")
v3 |> count(season) |> print(n = 20)

# Coverage flags summary
cat("\nCoverage flags:\n")
cat("  has_pfr (2018+):", sum(v3$has_pfr), "games\n")
cat("  has_ngs (2016+):", sum(v3$has_ngs), "games\n")
cat("  has_ftn (2022+):", sum(v3$has_ftn), "games\n")

# ============================================================
# 5. SAVE
# ============================================================
write_rds(v3, here("objects", "modeling_data_v3.rds"))
cat("\nSaved modeling_data_v3.rds:", nrow(v3), "rows x", ncol(v3), "columns\n")

# Save comparison
comparison <- tibble(
  version = c("v2", "v3"),
  rows = c(nrow(v2), nrow(v3)),
  cols = c(ncol(v2), ncol(v3)),
  new_features = c(NA, ncol(v3) - ncol(v2))
)
write_csv(comparison, here("results", "v2_v3_comparison.csv"))
cat("\nDone!\n")
