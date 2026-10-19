# Scenario search — discover on 2012-2019, confirm on 2020-2025
#
# MODE=recent: discover on 2018-2025 (edges fade, so weight recent years).
#   No holdout is left except 2026, so the guard against luck is
#   consistency: the scenario must beat breakeven in at least 6 of 8 seasons.
#
# Scenarios = one or two football conditions, betting the favorite or the
# underdog. Thresholds (terciles etc.) come from 2012-2019 only.
#   Discovery (2012-2019): fires >= 10 per season, wins >= 55%
#   Confirmation (2020-2025): games never used to find the scenario
#   Luck baseline: the whole search re-run with outcomes shuffled within season
#   2026: reported separately (fresh)

library(tidyverse); library(here)
set.seed(42)

v <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(!is.na(fav_cover), season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover"),
         rush_m = (fav_off_rush_epa_roll5 - dog_def_rush_epa_roll5) - (dog_off_rush_epa_roll5 - fav_def_rush_epa_roll5),
         pass_m = (fav_off_pass_epa_roll5 - dog_def_pass_epa_roll5) - (dog_off_pass_epa_roll5 - fav_def_pass_epa_roll5),
         run_over_pass = rush_m - pass_m,
         epa_delta = off_epa_play_roll5_diff - def_epa_play_roll5_diff,
         form_delta = (fav_off_epa_play_roll3 - fav_off_epa_play_roll5) - (dog_off_epa_play_roll3 - dog_off_epa_play_roll5),
         inj = rowMeans(cbind(scale(injury_diff)[, 1], -scale(fav_n_out)[, 1], scale(dog_n_out)[, 1]), na.rm = TRUE))

mode <- Sys.getenv("MODE", "split")
disc_seasons <- if (mode == "recent") 2018:2025 else 2012:2019
disc <- v |> filter(season %in% disc_seasons)
q <- function(x, p) quantile(disc[[x]], p, na.rm = TRUE)
yes <- function(x) coalesce(x, FALSE)

conds <- with(v, list(
  spread_le3       = yes(spread_fav <= 3),
  spread_3h_6h     = yes(spread_fav > 3 & spread_fav < 7),
  spread_7_9h      = yes(spread_fav >= 7 & spread_fav < 10),
  spread_10plus    = yes(spread_fav >= 10),
  fav_home         = fav_at_home == "yes",
  fav_road         = fav_at_home != "yes",
  division         = yes(div_game == 1),
  primetime        = yes(is_primetime == 1),
  fav_short_week   = fav_short_week == "yes",
  dog_short_week   = dog_short_week == "yes",
  fav_rest_edge    = yes(rest_diff >= 3),
  dog_rest_edge    = yes(rest_diff <= -3),
  bad_weather      = yes(weather_wind_mph >= 15 | weather_temperature <= 32),
  dome             = yes(is_dome == 1),
  high_total       = yes(over_under_line > 48),
  low_total        = yes(over_under_line < 41),
  early_season     = week <= 4,
  late_season      = week >= 13,
  elo_likes_fav    = yes(elo_spread_residual > 2),
  elo_likes_dog    = yes(elo_spread_residual < -2),
  epa_fav_strong   = yes(epa_delta >= q("epa_delta", 2/3)),
  epa_fav_weak     = yes(epa_delta <= q("epa_delta", 1/3)),
  fav_run_edge     = yes(run_over_pass >= q("run_over_pass", 2/3)),
  dog_run_edge     = yes(run_over_pass <= q("run_over_pass", 1/3)),
  dog_more_hurt    = yes(inj >= q("inj", 0.8)),
  fav_more_hurt    = yes(inj <= q("inj", 0.2)),
  fav_qb_change    = yes(fav_qb_change_flag == 1),
  dog_qb_change    = yes(dog_qb_change_flag == 1),
  fav_hot          = yes(form_delta >= q("form_delta", 2/3)),
  dog_hot          = yes(form_delta <= q("form_delta", 1/3)),
  fav_to_lucky     = yes(turnover_diff >= q("turnover_diff", 2/3)),
  dog_to_lucky     = yes(turnover_diff <= q("turnover_diff", 1/3))
))
cond_mat <- do.call(cbind, conds)

# All singles and pairs
pairs <- combn(names(conds), 2, simplify = FALSE)
scen_defs <- c(map(names(conds), ~ .x), pairs)
scen_names <- map_chr(scen_defs, ~ paste(.x, collapse = " & "))
scen_mat <- map(scen_defs, ~ apply(cond_mat[, .x, drop = FALSE], 1, all)) |> do.call(what = cbind)
colnames(scen_mat) <- scen_names

periods <- list(disc = v$season %in% disc_seasons, conf = v$season >= 2020 & v$season <= 2025, fresh = v$season == 2026)
n_seasons <- c(disc = 8, conf = 6)

evaluate <- function(y) {
  map_dfr(c("favorite", "underdog"), function(side) {
    win <- if (side == "favorite") y == 1 else y == 0
    stat <- function(p) {
      m <- scen_mat & periods[[p]]
      list(n = colSums(m), w = colSums(m & win))
    }
    d <- stat("disc"); c <- stat("conf"); f <- stat("fresh")
    # seasons in the discovery window where the scenario beat breakeven
    seasons_ok <- map(disc_seasons, function(s) {
      m <- scen_mat & (v$season == s); w <- colSums(m & win); n <- colSums(m)
      n > 0 & w / pmax(n, 1) > 0.524
    }) |> reduce(`+`)
    tibble(scenario = scen_names, side, seasons_ok,
           disc_n = d$n, disc_pct = d$w / d$n,
           conf_n = c$n, conf_pct = c$w / c$n,
           fresh_n = f$n, fresh_w = f$w)
  })
}

found <- function(res) {
  res <- res |> filter(disc_n >= 10 * n_seasons[["disc"]], disc_pct >= 0.55)
  if (mode == "recent") res <- res |> filter(seasons_ok >= 6)
  res
}

res <- evaluate(v$y)
hits <- found(res) |> arrange(desc(disc_pct))

# Luck baseline: shuffle outcomes within each season, repeat the search
shuffle_runs <- map_dfr(1:20, function(i) {
  ys <- v |> group_by(season) |> mutate(ys = sample(y)) |> pull(ys)
  h <- found(evaluate(ys))
  tibble(found = nrow(h), confirmed = sum(h$conf_n >= 10 * n_seasons[["conf"]] & h$conf_pct >= 0.55))
})

if (mode == "recent") {
  shuffled <- map_dbl(1:50, function(i) {
    ys <- v |> group_by(season) |> mutate(ys = sample(y)) |> pull(ys)
    nrow(found(evaluate(ys)))
  })
  cat(sprintf("Scenarios tested: %d (x2 sides) | discovery 2018-2025\n", length(scen_names)))
  cat(sprintf("Pass (10+/season, >=55%%, above breakeven in 6+ of 8 seasons): %d\n", nrow(hits)))
  cat(sprintf("Same search on shuffled outcomes: %.1f on average (95th pct %.0f)\n\n",
              mean(shuffled), quantile(shuffled, 0.95)))
  hits |> arrange(desc(seasons_ok), desc(disc_pct)) |>
    transmute(scenario, bet = side, per_season = round(disc_n / 8, 1),
              `2018-25` = sprintf("%.1f%% (%d)", 100 * disc_pct, disc_n),
              seasons_won = str_c(seasons_ok, "/8"),
              `2026 so far` = sprintf("%d-%d", fresh_w, fresh_n - fresh_w)) |>
    print(n = Inf, width = Inf)
  fresh_tot <- hits |> summarize(w = sum(fresh_w), n = sum(fresh_n))
  cat(sprintf("\nAll passing scenarios combined in 2026 so far: %d-%d (%.1f%%)\n",
              fresh_tot$w, fresh_tot$n - fresh_tot$w, 100 * fresh_tot$w / max(fresh_tot$n, 1)))
  write_rds(list(hits = hits, cuts_from = disc_seasons, shuffled_mean = mean(shuffled)),
            here("objects", "scenarios_recent.rds"))
  quit(save = "no")
}

cat(sprintf("Scenarios tested: %d (x2 sides)\n", length(scen_names)))
cat(sprintf("Found in 2012-2019 (10+/season, >=55%%): %d | with shuffled outcomes: %.0f on average\n",
            nrow(hits), mean(shuffle_runs$found)))
confirmed <- hits |> filter(conf_n >= 10 * n_seasons[["conf"]])
cat(sprintf("Of those, still firing 10+/season in 2020-2025: %d\n", nrow(confirmed)))
cat(sprintf("  ...and still winning >= 55%% in 2020-2025: %d | shuffled: %.1f on average\n",
            sum(confirmed$conf_pct >= 0.55), mean(shuffle_runs$confirmed)))
cat(sprintf("  ...and above breakeven (52.4%%): %d of %d\n", sum(confirmed$conf_pct > 0.524), nrow(confirmed)))
cat(sprintf("  Average 2020-2025 win rate of discovered scenarios: %.1f%%\n\n", 100 * weighted.mean(confirmed$conf_pct, confirmed$conf_n)))

cat("Top discovered scenarios and how they did on games they never saw:\n")
confirmed |> arrange(desc(conf_pct)) |>
  transmute(scenario, bet = side,
            `2012-19` = sprintf("%.1f%% (%d)", 100 * disc_pct, disc_n),
            `2020-25` = sprintf("%.1f%% (%d)", 100 * conf_pct, conf_n),
            `2026` = sprintf("%d-%d", fresh_w, fresh_n - fresh_w)) |>
  head(25) |> print(n = 25, width = Inf)

write_csv(res, here("results", "scenario_search_v37.csv"))
