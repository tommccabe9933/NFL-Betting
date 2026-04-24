# Era-Aware Feature Engineering
# Builds THREE era-specific datasets with NO zero-imputation for missing features.
# Era 1 (2012-2017): Base features only
# Era 2 (2018-2021): Base + PFR
# Era 3 (2022-2025): Base + PFR + NGS + FTN (full feature set)

library(tidyverse)
library(tidymodels)
library(here)

source(here("scripts", "14_research_protocol.R"))

v3 <- read_rds(here("objects", "modeling_data_v3.rds"))
cat("Loaded v3:", nrow(v3), "rows x", ncol(v3), "columns\n")

# ============================================================
# 1. DEFINE ERA-SPECIFIC FEATURE SETS
# ============================================================

# Base features available 2012+ (no premium data dependencies)
base_vars <- c(
  # Identifiers (excluded from modeling but needed for joins)
  "fav_cover", "game_id_norm", "season", "week",
  # Market
  "spread_fav", "over_under_line", "elo_diff",
  # Team efficiency (rolling)
  "ypp_diff", "pass_eff_diff", "rush_eff_diff", "turnover_diff",
  "pts_play_diff", "third_down_diff", "sack_rate_diff",
  "off_epa_diff", "def_epa_diff", "success_rate_diff",
  "fav_roll5_ypp", "dog_roll5_ypp", "fav_roll5_pts", "dog_roll5_pts",
  # Momentum
  "momentum_diff", "to_momentum_diff",
  # QB (from player stats, available 2012+)
  "qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
  # Situational EPA
  "pass_epa_mismatch", "rush_epa_mismatch",
  "third_down_epa_diff", "redzone_epa_diff",
  "explosive_diff", "neg_play_diff",
  # Injuries
  "injury_diff", "fav_qb_out", "dog_qb_out",
  # Context
  "div_game", "is_primetime", "is_dome",
  "rest_diff", "fav_at_home", "fav_short_week", "dog_short_week",
  "stadium_neutral",
  # Weather
  "weather_temperature", "weather_wind_mph",
  # New v3 engineered features
  "spread_squared", "abs_spread", "elo_spread_residual",
  "efficiency_vs_market", "wind_pass_interaction",
  "rest_advantage", "fav_rested_dog_short",
  "pass_rush_mismatch", "total_epa_edge"
)

# PFR features (2018+)
pfr_vars <- c(
  "fav_pfr_roll5_pressure_rate", "dog_pfr_roll5_pressure_rate",
  "fav_pfr_roll5_drop_rate", "dog_pfr_roll5_drop_rate",
  "fav_pfr_roll5_bad_throw_rate", "dog_pfr_roll5_bad_throw_rate",
  "fav_pfr_roll5_def_pressure", "dog_pfr_roll5_def_pressure"
)

# NGS features (2016+, but most reliable 2021+)
ngs_vars <- c(
  "fav_ngs_roll5_cpoe", "dog_ngs_roll5_cpoe",
  "fav_ngs_roll5_rush_yoe", "dog_ngs_roll5_rush_yoe",
  "fav_ngs_roll5_rush_eff", "dog_ngs_roll5_rush_eff",
  "fav_ngs_roll5_ttt", "dog_ngs_roll5_ttt",
  "fav_ngs_roll5_aggressiveness", "dog_ngs_roll5_aggressiveness"
)

# FTN features (2022+)
ftn_vars <- c(
  "fav_ftn_roll5_play_action", "dog_ftn_roll5_play_action",
  "fav_ftn_roll5_rpo", "dog_ftn_roll5_rpo",
  "fav_ftn_roll5_blitz", "dog_ftn_roll5_blitz",
  "fav_ftn_roll5_pass_rush", "dog_ftn_roll5_pass_rush"
)

# Python pipeline additional features (if available)
py_epa_vars <- intersect(names(v3), c(
  "fav_off_epa_play_roll5", "fav_off_pass_epa_roll5", "fav_off_rush_epa_roll5",
  "fav_off_success_rate_roll5", "fav_def_epa_play_roll5",
  "dog_off_epa_play_roll5", "dog_off_pass_epa_roll5", "dog_off_rush_epa_roll5",
  "dog_off_success_rate_roll5", "dog_def_epa_play_roll5",
  "off_epa_play_roll5_diff", "off_pass_epa_roll5_diff",
  "off_rush_epa_roll5_diff", "off_success_rate_roll5_diff",
  "def_epa_play_roll5_diff",
  # 3-game rolling (short-term momentum from Python)
  "fav_off_epa_play_roll3", "dog_off_epa_play_roll3",
  "off_epa_play_roll3_diff"
))

py_qb_vars <- intersect(names(v3), c(
  "fav_qb_epa_per_play_roll5", "dog_qb_epa_per_play_roll5",
  "fav_qb_cpoe_roll5", "dog_qb_cpoe_roll5",
  "fav_qb_sack_rate_roll5", "dog_qb_sack_rate_roll5",
  "fav_qb_change_flag", "dog_qb_change_flag",
  "fav_qb_out", "dog_qb_out",
  "qb_epa_per_play_roll5_diff", "qb_cpoe_roll5_diff"
))

py_sit_vars <- intersect(names(v3), c(
  "fav_off_3rd_epa_roll5", "dog_off_3rd_epa_roll5",
  "fav_off_rz_epa_roll5", "dog_off_rz_epa_roll5",
  "fav_off_explosive_rate_roll5", "dog_off_explosive_rate_roll5",
  "off_3rd_epa_roll5_diff", "off_rz_epa_roll5_diff"
))

py_injury_vars <- intersect(names(v3), c(
  "fav_injury_impact_score", "dog_injury_impact_score",
  "injury_impact_score_diff",
  "fav_n_out", "dog_n_out"
))

py_weather_vars <- intersect(names(v3), c(
  "temperature_f", "wind_mph", "humidity",
  "wind_over_12_hinge", "cold_game", "outdoor_flag",
  "wind_bin_0_5", "wind_bin_6_10", "wind_bin_11_15",
  "wind_bin_16_20", "wind_bin_20_plus"
))

py_ngs_vars <- intersect(names(v3), c(
  "fav_ngs_time_to_throw_roll5", "dog_ngs_time_to_throw_roll5",
  "fav_ngs_aggressiveness_roll5", "dog_ngs_aggressiveness_roll5",
  "fav_ngs_completed_air_yards_roll5", "dog_ngs_completed_air_yards_roll5",
  "fav_ngs_rush_yoe_roll5", "dog_ngs_rush_yoe_roll5"
))

# ============================================================
# 2. BUILD ERA-SPECIFIC DATASETS
# ============================================================

# Helper: select only columns that exist
safe_select <- function(data, vars) {
  existing <- intersect(vars, names(data))
  data |> select(all_of(existing))
}

# Era 1: 2012-2017 (base only)
era1_vars <- unique(c(base_vars, py_epa_vars, py_qb_vars, py_sit_vars,
                       py_injury_vars, py_weather_vars))
era1_data <- v3 |>
  filter(season >= 2012, season <= 2017) |>
  safe_select(era1_vars)

cat("\nEra 1 (2012-2017, base):", nrow(era1_data), "games x",
    ncol(era1_data), "columns\n")

# Era 2: 2018-2021 (base + PFR)
era2_vars <- unique(c(base_vars, pfr_vars, py_epa_vars, py_qb_vars,
                       py_sit_vars, py_injury_vars, py_weather_vars))
era2_data <- v3 |>
  filter(season >= 2018, season <= 2021) |>
  safe_select(era2_vars)

cat("Era 2 (2018-2021, base+PFR):", nrow(era2_data), "games x",
    ncol(era2_data), "columns\n")

# Era 3: 2022-2025 (base + PFR + NGS + FTN)
era3_vars <- unique(c(base_vars, pfr_vars, ngs_vars, ftn_vars,
                       py_epa_vars, py_qb_vars, py_sit_vars,
                       py_injury_vars, py_weather_vars, py_ngs_vars))
era3_data <- v3 |>
  filter(season >= 2022) |>
  safe_select(era3_vars)

cat("Era 3 (2022-2025, full):", nrow(era3_data), "games x",
    ncol(era3_data), "columns\n")

# Full dataset with coverage flags (for models that handle missingness)
full_data <- v3

# ============================================================
# 3. VALIDATE NO ZERO-IMPUTATION ARTIFACTS
# ============================================================
cat("\n=== VALIDATION: No hidden era flags ===\n")

# In era1, PFR/NGS/FTN columns should NOT exist (not be zero-imputed)
for (v in c(pfr_vars[1], ngs_vars[1], ftn_vars[1])) {
  if (v %in% names(era1_data)) {
    cat("  WARNING:", v, "exists in era1 — should not be there!\n")
  } else {
    cat("  OK:", v, "correctly excluded from era1\n")
  }
}

# In era3, all columns should have <50% NA
if (ncol(era3_data) > 5) {
  na_pcts <- map_dbl(era3_data |> select(where(is.numeric)), ~ mean(is.na(.x)))
  high_na <- names(na_pcts[na_pcts > 0.5])
  if (length(high_na) > 0) {
    cat("  WARNING: High NA columns in era3:", paste(high_na, collapse=", "), "\n")
  } else {
    cat("  OK: All era3 numeric columns have <50% NA\n")
  }
}

# ============================================================
# 4. SAVE
# ============================================================
write_rds(era1_data, here("objects", "era1_data.rds"))
write_rds(era2_data, here("objects", "era2_data.rds"))
write_rds(era3_data, here("objects", "era3_data.rds"))

cat("\nSaved era-specific datasets:\n")
cat("  era1_data.rds:", nrow(era1_data), "x", ncol(era1_data), "\n")
cat("  era2_data.rds:", nrow(era2_data), "x", ncol(era2_data), "\n")
cat("  era3_data.rds:", nrow(era3_data), "x", ncol(era3_data), "\n")

cat("\nDone! Ready for Phase 3 modeling.\n")
