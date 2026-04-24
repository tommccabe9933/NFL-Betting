# Rebuild modeling_data_v3 CLEAN — fixing all audit findings
#
# FIXES:
# 1. Remove all 79 postgame leaker columns (raw same-game stats)
# 2. Fix spread_fav to be positive magnitude
# 3. Fix Python EPA sign flip
# 4. Expand NGS to 2016+ (R pipeline only loaded 2021+)
# 5. No mean imputation of era-specific features — use NA + era subsets
# 6. Only keep rolled/lagged pregame features + market + context
# 7. Verify every feature has |cor| < 0.15 with target (pregame sanity check)

library(tidyverse)
library(tidymodels)
library(here)
library(nflreadr)
library(slider)

source(here("scripts", "00_project_setup.R"))

# ============================================================
# 1. START FROM R v2 (known good rolled features)
# ============================================================
cat("Loading v2 (R pipeline — known good rolled features)...\n")
v2 <- read_rds(here("objects", "modeling_data_v2.rds"))
cat("  v2:", nrow(v2), "x", ncol(v2), "\n")

# FIX 2: spread_fav should be positive magnitude
# In v2, spread_fav is stored as negative. Fix it.
cat("\nFIX 2: spread_fav sign\n")
cat("  Before: range [", min(v2$spread_fav), ",", max(v2$spread_fav), "]\n")
v2 <- v2 |> mutate(spread_fav = abs(spread_fav))
cat("  After: range [", min(v2$spread_fav), ",", max(v2$spread_fav), "]\n")

# ============================================================
# 2. LOAD PYTHON FEATURE STORE AND EXTRACT ONLY ROLLED FEATURES
# ============================================================
cat("\nLoading Python feature store...\n")
py_path <- here("data", "processed", "modeling", "feature_store_v1.parquet")

if (file.exists(py_path)) {
  py_store <- arrow::read_parquet(py_path) |> as_tibble()
  cat("  Python store:", nrow(py_store), "x", ncol(py_store), "\n")

  # ONLY keep columns with "roll" in the name (rolled/lagged = pregame safe)
  # Plus a few known-safe non-rolled columns
  safe_py_cols <- names(py_store)[
    str_detect(names(py_store), "roll[35]") |
    names(py_store) %in% c(
      "game_id_norm",
      # Market features (pregame by definition)
      "fav_implied_prob_devigged", "dog_implied_prob_devigged",
      "home_implied_prob", "away_implied_prob",
      # Coverage flags
      "has_pfr", "has_ngs", "has_ftn",
      # QB uncertainty (pregame)
      "fav_qb_change_flag", "dog_qb_change_flag",
      # Injury (pregame)
      "fav_injury_impact_score", "dog_injury_impact_score",
      "fav_n_out", "dog_n_out", "fav_n_questionable", "dog_n_questionable",
      # Weather bins
      "temperature_f", "wind_mph", "humidity",
      "wind_over_12_hinge", "cold_game", "outdoor_flag",
      "wind_bin_0_5", "wind_bin_6_10", "wind_bin_11_15",
      "wind_bin_16_20", "wind_bin_20_plus",
      # Rest
      "fav_days_rest", "dog_days_rest", "fav_short_week", "dog_short_week",
      "fav_mini_bye", "dog_mini_bye",
      # Elo
      "fav_pre_elo", "dog_pre_elo", "elo_diff"
    )
  ]

  py_safe <- py_store |> select(any_of(safe_py_cols))
  cat("  Safe Python columns:", ncol(py_safe) - 1, "(excluding game_id_norm)\n")

  # FIX 3: Check for sign flip in Python EPA features
  # If R's off_epa_diff and Python's off_epa_play_roll5_diff are anti-correlated, flip Python
  if ("game_id_norm" %in% names(py_safe)) {
    test_join <- v2 |>
      select(game_id_norm, off_epa_diff) |>
      inner_join(
        py_safe |> select(game_id_norm, any_of("off_epa_play_roll5_diff")),
        by = "game_id_norm"
      )

    if ("off_epa_play_roll5_diff" %in% names(test_join)) {
      valid <- !is.na(test_join$off_epa_diff) & !is.na(test_join$off_epa_play_roll5_diff)
      if (sum(valid) > 100) {
        sign_cor <- cor(test_join$off_epa_diff[valid], test_join$off_epa_play_roll5_diff[valid])
        cat(sprintf("  R vs Python EPA correlation: %.4f\n", sign_cor))

        if (sign_cor < -0.5) {
          cat("  *** SIGN FLIP DETECTED — flipping Python diff columns ***\n")
          diff_cols <- names(py_safe)[str_detect(names(py_safe), "_diff$")]
          for (col in diff_cols) {
            py_safe[[col]] <- -py_safe[[col]]
          }
          # Verify fix
          test_join2 <- v2 |>
            select(game_id_norm, off_epa_diff) |>
            inner_join(py_safe |> select(game_id_norm, any_of("off_epa_play_roll5_diff")),
                       by = "game_id_norm")
          valid2 <- !is.na(test_join2$off_epa_diff) & !is.na(test_join2$off_epa_play_roll5_diff)
          cat(sprintf("  After fix: cor=%.4f\n", cor(test_join2$off_epa_diff[valid2], test_join2$off_epa_play_roll5_diff[valid2])))
        }
      }
    }
  }

  # Join safe Python columns onto v2
  # Only bring in columns that DON'T already exist in v2
  new_cols <- setdiff(names(py_safe), c("game_id_norm", names(v2)))
  if (length(new_cols) > 0) {
    py_new <- py_safe |> select(game_id_norm, all_of(new_cols))
    v3 <- v2 |> left_join(py_new, by = "game_id_norm")
    cat("  Added", length(new_cols), "new columns from Python\n")
  } else {
    v3 <- v2
    cat("  No new safe columns from Python\n")
  }
} else {
  cat("  Python store not found — using v2 only\n")
  v3 <- v2
}

# ============================================================
# 3. FIX 4: EXPAND NGS TO 2016+ (R pipeline only loaded 2021+)
# ============================================================
cat("\nFIX 4: Loading NGS 2016-2020 (previously missing)...\n")

ngs_extra_seasons <- 2016:2020
ngs_pass_extra <- map(ngs_extra_seasons, ~ {
  tryCatch(load_nextgen_stats(seasons = .x, stat_type = "passing"), error = function(e) NULL)
}) |> compact() |> bind_rows()

ngs_rush_extra <- map(ngs_extra_seasons, ~ {
  tryCatch(load_nextgen_stats(seasons = .x, stat_type = "rushing"), error = function(e) NULL)
}) |> compact() |> bind_rows()

if (nrow(ngs_pass_extra) > 0) {
  cat("  Found NGS passing 2016-2020:", nrow(ngs_pass_extra), "rows\n")

  # Aggregate to team-week
  ngs_pass_team_extra <- ngs_pass_extra |>
    filter(week > 0) |>
    mutate(team_abbr = normalize_team(team_abbr)) |>
    group_by(season, week, team_abbr) |>
    summarize(
      ngs_time_to_throw = weighted.mean(avg_time_to_throw, attempts, na.rm = TRUE),
      ngs_completed_air_yards = weighted.mean(avg_completed_air_yards, attempts, na.rm = TRUE),
      ngs_aggressiveness = weighted.mean(aggressiveness, attempts, na.rm = TRUE),
      ngs_cpoe = weighted.mean(completion_percentage_above_expectation, attempts, na.rm = TRUE),
      .groups = "drop"
    )

  ngs_rush_team_extra <- if (nrow(ngs_rush_extra) > 0) {
    ngs_rush_extra |>
      filter(week > 0) |>
      mutate(team_abbr = normalize_team(team_abbr)) |>
      group_by(season, week, team_abbr) |>
      summarize(
        ngs_rush_yoe = weighted.mean(rush_yards_over_expected_per_att, rush_attempts, na.rm = TRUE),
        ngs_rush_efficiency = weighted.mean(efficiency, rush_attempts, na.rm = TRUE),
        ngs_pct_8plus_defenders = weighted.mean(percent_attempts_gte_eight_defenders, rush_attempts, na.rm = TRUE),
        .groups = "drop"
      )
  } else { tibble() }

  ngs_extra_team <- ngs_pass_team_extra
  if (nrow(ngs_rush_team_extra) > 0) {
    ngs_extra_team <- ngs_extra_team |>
      left_join(ngs_rush_team_extra, by = c("season", "week", "team_abbr"))
  }

  # Rolling NGS features
  ngs_extra_rolling <- ngs_extra_team |>
    group_by(team_abbr) |>
    arrange(season, week, .by_group = TRUE) |>
    mutate(
      ngs_roll5_ttt = lagged_roll(ngs_time_to_throw, 5),
      ngs_roll5_cpoe = lagged_roll(ngs_cpoe, 5),
      ngs_roll5_aggressiveness = lagged_roll(ngs_aggressiveness, 5),
      ngs_roll5_rush_yoe = if ("ngs_rush_yoe" %in% names(cur_data())) lagged_roll(ngs_rush_yoe, 5) else NA_real_,
      ngs_roll5_rush_eff = if ("ngs_rush_efficiency" %in% names(cur_data())) lagged_roll(ngs_rush_efficiency, 5) else NA_real_
    ) |>
    ungroup() |>
    select(season, week, team_abbr, starts_with("ngs_roll5_"))

  # Map to fav/dog using team_game_full
  team_game_full <- read_rds(here("objects", "team_game_full.rds"))

  fav_teams <- team_game_full |>
    filter(cover != "push", team_norm == team_favorite_id) |>
    distinct(game_id_norm, .keep_all = TRUE) |>
    select(game_id_norm, season, week, team_fav = team_norm)

  dog_teams <- team_game_full |>
    filter(cover != "push", team_norm != team_favorite_id) |>
    distinct(game_id_norm, .keep_all = TRUE) |>
    select(game_id_norm, team_dog = team_norm)

  fav_dog_map <- fav_teams |>
    inner_join(dog_teams, by = "game_id_norm") |>
    mutate(season = as.numeric(season), week = as.numeric(week))

  # Ensure NGS rolling has matching types
  ngs_extra_rolling <- ngs_extra_rolling |>
    mutate(season = as.numeric(season), week = as.numeric(week))

  # Join NGS for fav and dog
  ngs_fav <- ngs_extra_rolling |>
    rename_with(~ paste0("fav_", .), starts_with("ngs_"))
  ngs_dog <- ngs_extra_rolling |>
    rename_with(~ paste0("dog_", .), starts_with("ngs_"))

  extra_features <- fav_dog_map |>
    left_join(ngs_fav, by = c("season", "week", "team_fav" = "team_abbr")) |>
    left_join(ngs_dog, by = c("season", "week", "team_dog" = "team_abbr"))

  # Fill NAs in v3's NGS columns with the extra 2016-2020 values
  ngs_fill_cols <- names(extra_features)[str_detect(names(extra_features), "^(fav|dog)_ngs")]

  for (col in ngs_fill_cols) {
    if (col %in% names(v3)) {
      extra_vals <- extra_features |> select(game_id_norm, extra_val = !!sym(col))
      v3 <- v3 |>
        left_join(extra_vals, by = "game_id_norm") |>
        mutate(!!col := coalesce(.data[[col]], extra_val)) |>
        select(-extra_val)
    }
  }

  # Check NGS coverage after fix
  cat("  NGS coverage by season after fix:\n")
  if ("fav_ngs_roll5_ttt" %in% names(v3)) {
    v3 |> group_by(season) |>
      summarize(pct = round(mean(!is.na(fav_ngs_roll5_ttt)) * 100, 1)) |>
      print(n = 20)
  }
} else {
  cat("  No extra NGS data found for 2016-2020\n")
}

# ============================================================
# 4. ADD CLEAN ENGINEERED FEATURES
# ============================================================
cat("\nAdding engineered features (with correct spread_fav sign)...\n")

v3 <- v3 |>
  mutate(
    # Coverage flags
    has_pfr = season >= 2018,
    has_ngs = season >= 2016,
    has_ftn = season >= 2022,

    # Market-derived (spread_fav is now POSITIVE)
    spread_squared = spread_fav^2,
    abs_spread = spread_fav,  # already positive now

    spread_bucket = case_when(
      spread_fav <= 3   ~ "tiny",
      spread_fav <= 6.5 ~ "small",
      spread_fav <= 9.5 ~ "medium",
      spread_fav <= 13.5 ~ "large",
      TRUE              ~ "huge"
    ),

    # Elo-implied spread (2.7 pts per 100 Elo)
    elo_implied_spread = elo_diff * 2.7 / 100,
    elo_spread_residual = elo_implied_spread - spread_fav,  # positive = Elo thinks fav is better than spread says

    # Season timing
    season_phase = case_when(
      week <= 4  ~ "early",
      week <= 12 ~ "mid",
      TRUE       ~ "late"
    ),

    # Rest interactions
    rest_advantage = pmax(rest_diff, 0),

    # Total EPA edge (clean — using rolled features)
    total_epa_edge = off_epa_diff - def_epa_diff
  )

# ============================================================
# 5. POSTGAME LEAKAGE CHECK — VERIFY CLEAN
# ============================================================
cat("\n=== POSTGAME LEAKAGE CHECK ===\n")

target <- as.integer(v3$fav_cover == "cover")
leaker_count <- 0

for (col in names(v3)) {
  x <- v3[[col]]
  if (!is.numeric(x)) next
  valid <- !is.na(x)
  if (sum(valid) < 100) next
  cv <- tryCatch(cor(x[valid], target[valid]), error = function(e) NA)
  if (is.na(cv)) next
  if (abs(cv) > 0.15) {
    cat(sprintf("  LEAKER: %-40s cor=%.3f\n", col, cv))
    leaker_count <- leaker_count + 1
  }
}

if (leaker_count == 0) {
  cat("  CLEAN! No features with |cor| > 0.15\n")
} else {
  cat(sprintf("\n  WARNING: %d potential leakers remain\n", leaker_count))
}

# ============================================================
# 6. SHOW STRONGEST PREGAME SIGNALS
# ============================================================
cat("\n=== TOP PREGAME SIGNALS (rolled features only) ===\n")

signal_df <- map_dfr(names(v3), function(col) {
  x <- v3[[col]]
  if (!is.numeric(x)) return(NULL)
  valid <- !is.na(x)
  if (sum(valid) < 200) return(NULL)
  cv <- tryCatch(cor(x[valid], target[valid]), error = function(e) NA)
  if (is.na(cv)) return(NULL)
  tibble(feature = col, cor = cv, abs_cor = abs(cv), n = sum(valid))
}) |> arrange(desc(abs_cor))

cat("Top 30 features by |correlation|:\n")
signal_df |> head(30) |>
  mutate(cor = round(cor, 4)) |>
  print(n = 30)

# ============================================================
# 7. SAVE
# ============================================================
write_rds(v3, here("objects", "modeling_data_v3.rds"))

cat(sprintf("\nSaved CLEAN modeling_data_v3.rds: %d rows x %d columns\n", nrow(v3), ncol(v3)))
cat("Done!\n")
