# Combo scenarios — situation (recent results, ATS streaks, rest) x matchup
#
# Part A: Tom's pre-registered rule, tested on its own:
#   favorite has NOT covered in its last 3 games, AND its run offense has a
#   big edge over the opponent's run defense (top third, 2018-2025 cutoff)
#   -> bet the FAVORITE (market sours on a slumping team that still has a
#      matchup advantage)
# Part B: every situation x matchup pair, both sides, with the same bar as
#   before (5+/season, >=55% in 2018-25, above breakeven in 6+ of 8 seasons)
#   and a shuffled-outcome luck baseline.

library(tidyverse); library(here)
source(here("scripts", "scenarios.R"))
source(here("scripts", "schedule_features.R"))
set.seed(42)

g <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(!is.na(fav_cover), season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover")) |>
  add_schedule_features()

cuts <- learn_combo_cuts(g, 2018:2025)
all_conds <- combo_conditions(g, cuts)
situ_names <- c("fav_no_cover_last3", "fav_covered_last3", "dog_no_cover_last3", "dog_covered_last3",
                "fav_lost_last", "fav_upset_last", "dog_won_last", "dog_upset_win_last",
                "fav_blowout_win", "dog_blowout_loss", "fav_off_bye", "dog_off_bye", "fav_short_rest")
situ <- all_conds[situ_names]
stat <- all_conds[setdiff(names(all_conds), situ_names)]

record <- function(hit, side, y = g$y) {
  win <- if (side == "favorite") y == 1 else y == 0
  per <- map_dfr(2018:2025, ~ tibble(n = sum(hit & g$season == .x), w = sum(hit & win & g$season == .x)))
  r <- hit & g$season %in% 2018:2025; o <- hit & g$season %in% 2012:2017; f <- hit & g$season == 2026
  tibble(per_season = sum(r) / 8, win_2018_25 = sum(r & win) / sum(r),
         seasons_won = sum(per$n > 0 & per$w / pmax(per$n, 1) > 0.524),
         win_2012_17 = sum(o & win) / sum(o), fresh_w = sum(f & win), fresh_n = sum(f))
}

# ---------------- Part A ----------------
cat("=== PART A: your rule (pre-registered, single test) ===\n")
part_a <- bind_rows(
  record(situ$fav_no_cover_last3 & stat$fav_rush_mismatch, "favorite") |> mutate(rule = "No cover last 3 + big run edge -> favorite"),
  record(situ$fav_no_cover_last3, "favorite") |> mutate(rule = "  (no cover last 3 alone)"),
  record(stat$fav_rush_mismatch, "favorite") |> mutate(rule = "  (big run edge alone)"))
part_a |> transmute(rule, per_season = round(per_season, 1), `2018-25` = sprintf("%.1f%%", 100 * win_2018_25),
                    seasons_won = str_c(seasons_won, "/8"), `2012-17` = sprintf("%.1f%%", 100 * win_2012_17),
                    `2026` = sprintf("%d-%d", fresh_w, fresh_n - fresh_w)) |> print(width = Inf)

# ---------------- Part B ----------------
combos <- expand_grid(s = names(situ), m = names(stat), side = c("favorite", "underdog"))
eval_all <- function(y) pmap_dfr(combos, function(s, m, side)
  record(situ[[s]] & stat[[m]], side, y) |> mutate(scenario = str_c(s, " & ", m), bet = side))
pass <- function(r) r |> filter(per_season >= 5, win_2018_25 >= 0.55, seasons_won >= 6)

res <- eval_all(g$y)
shuffled <- map_dbl(1:40, function(i) {
  ys <- g |> group_by(season) |> mutate(ys = sample(y)) |> pull(ys)
  nrow(pass(eval_all(ys)))
})
hits <- pass(res)

cat(sprintf("\n=== PART B: %d situation x matchup combos (x2 sides) ===\n", nrow(combos) / 2))
cat(sprintf("Pass the bar: %d | shuffled outcomes: %.1f on average (95th pct %.0f)\n\n",
            nrow(hits), mean(shuffled), quantile(shuffled, 0.95)))
hits |> arrange(desc(seasons_won), desc(win_2018_25)) |>
  transmute(scenario, bet, per_season = round(per_season, 1), `2018-25` = sprintf("%.1f%%", 100 * win_2018_25),
            seasons_won = str_c(seasons_won, "/8"), `2012-17` = sprintf("%.1f%%", 100 * win_2012_17),
            `2026` = sprintf("%d-%d", fresh_w, fresh_n - fresh_w)) |>
  print(n = Inf, width = Inf)
write_csv(res, here("results", "combo_scenarios_v37.csv"))

# ---------------- Independent check: 2012-2017 (not used to find anything) ----------------
also_old <- function(r) pass(r) |> filter(win_2012_17 >= 0.55)
shuffled_old <- map_dbl(1:40, function(i) {
  ys <- g |> group_by(season) |> mutate(ys = sample(y)) |> pull(ys)
  nrow(also_old(eval_all(ys)))
})
cat(sprintf("\nAlso >= 55%% in 2012-2017: %d of %d | shuffled outcomes: %.1f on average (95th pct %.0f)\n",
            nrow(also_old(res)), nrow(hits), mean(shuffled_old), quantile(shuffled_old, 0.95)))
also_old(res) |> arrange(desc(win_2012_17)) |>
  transmute(scenario, bet, per_season = round(per_season, 1), `2018-25` = sprintf("%.1f%%", 100 * win_2018_25),
            seasons_won = str_c(seasons_won, "/8"), `2012-17` = sprintf("%.1f%%", 100 * win_2012_17),
            `2026` = sprintf("%d-%d", fresh_w, fresh_n - fresh_w)) |>
  print(n = Inf, width = Inf)
write_rds(list(rules = also_old(res), cuts = cuts, frozen_on = as.character(Sys.Date())),
          here("objects", "combo_scenarios_two_era.rds"))
