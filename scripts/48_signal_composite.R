# Signal composite — a deliberately simple model built from the lessons of v3
#
# Lessons carried over: the market prices team quality, so look for small,
# *stable* signals; average many weak signals instead of fitting hundreds of
# coefficients; abstain unless the signal is strong; judge only out of sample.
#
# Protocol (no peeking):
#   1. Feature selection uses only seasons before the one being predicted.
#      A feature qualifies if its correlation with covering points the same
#      way in nearly every season (sign stability), not just on average.
#   2. Settings (number of features K, betting cutoff) are chosen once, on
#      2021-2022, with features selected on 2016-2020.
#   3. Walk-forward test on 2023-2025 with those settings frozen.
# Caveat: 2025 has been examined in earlier rounds, so 2026 is the only
# fully clean test.

library(tidyverse); library(here)

data_version <- Sys.getenv("MODEL_VERSION", "v37")
v <- read_rds(here("objects", str_glue("modeling_data_{data_version}.rds")))
played <- v |> filter(!is.na(fav_cover)) |> mutate(y = as.integer(fav_cover == "cover"))

id_cols <- c("season", "week", "y")
candidates <- played |> select(where(is.numeric), -any_of(id_cols)) |> names()

# Per-season correlation of each feature with covering
season_cors <- function(df, seasons) {
  df <- df |> filter(season %in% seasons)
  map_dfr(candidates, function(f) {
    by_s <- df |> group_by(season) |>
      summarize(n = sum(!is.na(.data[[f]])),
                r = if (n > 50 && sd(.data[[f]], na.rm = TRUE) > 0)
                      cor(.data[[f]], y, use = "complete.obs") else NA_real_,
                .groups = "drop")
    tibble(feature = f, mean_r = mean(by_s$r, na.rm = TRUE),
           seasons_ok = sum(!is.na(by_s$r)),
           same_sign = mean(sign(by_s$r) == sign(mean(by_s$r, na.rm = TRUE)), na.rm = TRUE))
  })
}

# Select up to K stable, non-redundant features from `seasons`
select_features <- function(df, seasons, k) {
  ranked <- season_cors(df, seasons) |>
    filter(seasons_ok >= length(seasons) - 1, same_sign >= 0.8, abs(mean_r) >= 0.02) |>
    arrange(desc(abs(mean_r)))
  train <- df |> filter(season %in% seasons)
  picked <- character()
  for (f in ranked$feature) {
    if (length(picked) >= k) break
    redundant <- any(map_lgl(picked, ~ abs(cor(train[[f]], train[[.x]], use = "pairwise.complete.obs")) > 0.6))
    if (!redundant) picked <- c(picked, f)
  }
  ranked |> filter(feature %in% picked) |>
    mutate(mu = map_dbl(feature, ~ mean(train[[.x]], na.rm = TRUE)),
           sd = map_dbl(feature, ~ sd(train[[.x]], na.rm = TRUE)))
}

# Composite = average signed z-score (missing features skipped), re-scaled
# to unit SD on the training seasons
score <- function(df, sel, train_df) {
  raw <- function(d) {
    z <- map(seq_len(nrow(sel)), ~ sign(sel$mean_r[.x]) * (d[[sel$feature[.x]]] - sel$mu[.x]) / sel$sd[.x])
    rowMeans(do.call(cbind, z), na.rm = TRUE)
  }
  tr <- raw(train_df)
  (raw(df) - mean(tr, na.rm = TRUE)) / sd(tr, na.rm = TRUE)
}

evaluate <- function(df, s, cutoff) {
  df |> mutate(s = s,
               bet = case_when(s >= cutoff ~ "cover", s <= -cutoff ~ "no_cover", TRUE ~ "pass"),
               win = (bet == "cover" & y == 1) | (bet == "no_cover" & y == 0)) |>
    filter(bet != "pass")
}

summ <- function(b) b |> summarize(bets = n(), win_rate = mean(win),
                                   profit = sum(if_else(win, 100, -110)))

# ---------------------------------------------------------------
# Step 1-2: choose K and cutoff on 2021-2022 (features from 2016-2020)
# ---------------------------------------------------------------
grid <- expand_grid(k = c(5, 10, 15, 25), cutoff = c(0.5, 0.75, 1.0, 1.25))
design <- 2016:2020
sel_by_k <- map(set_names(unique(grid$k)), ~ select_features(played, design, .x))
val <- played |> filter(season %in% 2021:2022)
tuning <- grid |> mutate(res = map2(k, cutoff, function(k, cutoff) {
  sel <- sel_by_k[[as.character(k)]]
  summ(evaluate(val, score(val, sel, played |> filter(season %in% design)), cutoff))
})) |> unnest(res) |> filter(bets >= 60) |> arrange(desc(profit))

cat("=== TUNING on 2021-2022 (features picked on 2016-2020) ===\n")
print(tuning |> mutate(win_rate = round(win_rate, 3)), n = Inf)
best <- tuning |> slice(1)
cat(sprintf("\nChosen: K = %d features, bet when |score| >= %.2f SD\n", best$k, best$cutoff))

# ---------------------------------------------------------------
# Step 3: walk-forward test 2023-2025, settings frozen
# Features re-selected each year on 2016 .. (test season - 1)
# ---------------------------------------------------------------
test_bets <- map_dfr(2023:2025, function(ts) {
  seasons <- 2016:(ts - 1)
  sel <- select_features(played, seasons, best$k)
  te <- played |> filter(season == ts)
  evaluate(te, score(te, sel, played |> filter(season %in% seasons)), best$cutoff) |>
    mutate(test_season = ts)
})

cat("\n=== WALK-FORWARD TEST 2023-2025 ===\n")
test_bets |> group_by(test_season) |> summ() |> print()
tot <- summ(test_bets)
cat(sprintf("TOTAL: %d bets, %.1f%%, $%d | binomial p vs 52.4%%: %.3f\n",
  tot$bets, 100 * tot$win_rate, round(tot$profit),
  binom.test(sum(test_bets$win), nrow(test_bets), 0.524, alternative = "greater")$p.value))
test_bets |> group_by(bet) |> summ() |> print()

# ---------------------------------------------------------------
# Live: select on 2016-2025, score upcoming games
# ---------------------------------------------------------------
live_seasons <- 2016:max(played$season[played$season < max(v$season)])
sel_live <- select_features(played, live_seasons, best$k)
cat("\nFeatures in the live composite (sign = direction that favors the favorite covering):\n")
print(sel_live |> transmute(feature, mean_r = round(mean_r, 3), same_sign = round(same_sign, 2)), n = Inf)

up <- v |> filter(is_upcoming)
upcoming <- read_rds(here("objects", "upcoming_games.rds"))
picks <- up |>
  mutate(score = score(up, sel_live, played |> filter(season %in% live_seasons))) |>
  left_join(upcoming |> select(game_id_norm, gameday, gametime, home_team, away_team, spread_line), by = "game_id_norm") |>
  filter(as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York") > Sys.time()) |>
  mutate(favorite = if_else(spread_line >= 0, home_team, away_team),
         underdog = if_else(spread_line >= 0, away_team, home_team),
         pick = case_when(score >= best$cutoff ~ str_glue("{favorite} -{spread_fav}"),
                          score <= -best$cutoff ~ str_glue("{underdog} +{spread_fav}"),
                          TRUE ~ "no bet"),
         matchup = str_glue("{away_team} @ {home_team}")) |>
  arrange(gameday, gametime) |>
  transmute(week, gameday, gametime, matchup, line = spread_fav, pick, score = round(score, 2))

out <- here("results", str_glue("picks_{max(v$season)}_wk{str_pad(max(picks$week), 2, pad = '0')}_composite.csv"))
write_csv(picks, out)
cat("\n=== COMPOSITE PICKS ===\n"); print(picks, n = Inf)
write_rds(list(settings = best, features = sel_live, test_bets = test_bets),
          here("objects", "signal_composite.rds"))
