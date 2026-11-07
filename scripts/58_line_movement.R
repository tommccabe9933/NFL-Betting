# Line movement — opening vs closing spreads (Kaggle: archangel0x01/nfl-sportsbook-odds-and-results)
#
# Opener = first consensus spread quoted within 8 days of kickoff (2018-2022
# line history; earlier quotes are summer lookahead lines). For 2023+ the
# ESPN BET / DraftKings published open is used. Close = last pre-game quote
# (or the book's published close).
#
# Pre-registered tests:
#   A. Steam: after a 1+ point move toward a side, does that side cover the close?
#   B. Key numbers: moves through 3 or 7
#   C. Closing line value of our rule sets: does the line move TOWARD the side
#      our scenarios/combos pick? Positive CLV = the market later agrees.

library(tidyverse); library(here); library(arrow)
source(here("scripts", "00_project_setup.R"))
source(here("scripts", "scenarios.R")); source(here("scripts", "schedule_features.R"))

d <- here("data", "raw", "kaggle_odds")
games <- read_parquet(file.path(d, "games.parquet")) |> filter(season_type == "regular")
hist <- read_parquet(file.path(d, "line_history.parquet")) |> filter(market == "spread")
books <- read_parquet(file.path(d, "bookmaker_odds.parquet")) |> filter(market == "spread")

# Opener / close from consensus line history (2018-2022)
lh <- hist |> inner_join(games |> select(game_id, start_time_utc), by = "game_id") |>
  filter(quote_time_utc <= start_time_utc,
         difftime(start_time_utc, quote_time_utc, units = "hours") <= 192) |>
  arrange(game_id, quote_time_utc) |>
  group_by(game_id) |> summarize(open_home = first(line), close_home = last(line), src = "consensus history", .groups = "drop")

# Published open/close for games without history (2023+), preferring ESPN BET then DraftKings
pub <- books |> filter(price_type %in% c("open", "close"), bookmaker %in% c("ESPN BET", "DraftKings")) |>
  select(game_id, bookmaker, price_type, line) |>
  pivot_wider(names_from = price_type, values_from = line) |>
  filter(!is.na(open), !is.na(close)) |>
  mutate(pref = match(bookmaker, c("ESPN BET", "DraftKings"))) |>
  group_by(game_id) |> slice_min(pref, n = 1, with_ties = FALSE) |> ungroup() |>
  transmute(game_id, open_home = open, close_home = close, src = bookmaker)

lines <- bind_rows(lh, pub |> filter(!(game_id %in% lh$game_id)))

name_to_abbr <- function(x) normalize_team(team_lookup$team[match(x, team_lookup$team_name)])
odds <- games |>
  transmute(game_id, season, week, home = name_to_abbr(home_team), away = name_to_abbr(away_team),
            game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", away, "_", home)) |>
  inner_join(lines, by = "game_id")
cat(sprintf("Games with open + close: %d (unmatched team names: %d)\n",
            nrow(odds), sum(is.na(odds$home) | is.na(odds$away))))

v <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(!is.na(fav_cover), season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover"))
m <- v |> inner_join(odds |> select(game_id_norm, open_home, close_home, src), by = "game_id_norm") |>
  mutate(fav_home = fav_at_home == "yes",
         # positive = line moved toward the favorite (favorite now lays more points)
         toward_fav = if_else(fav_home, open_home - close_home, close_home - open_home),
         open_fav = abs(open_home), close_fav = abs(close_home))
cat(sprintf("Matched to modeling table: %d games (%s)\n\n", nrow(m),
            paste(names(table(m$season)), table(m$season), sep = ":", collapse = " ")))

rec <- function(win) sprintf("%d-%d (%.1f%%)", sum(win), sum(!win), 100 * mean(win))
by_era <- function(d, win_expr) {
  d <- d |> mutate(win = {{ win_expr }})
  tibble(`2018-22` = rec(d$win[d$season <= 2022]), `2023-26` = rec(d$win[d$season >= 2023]), all = rec(d$win))
}

cat("=== A. STEAM: bet the side the line moved toward, at the close ===\n")
bind_rows(
  by_era(filter(m, toward_fav >= 1), y == 1) |> mutate(test = "Moved 1+ toward favorite -> favorite"),
  by_era(filter(m, toward_fav <= -1), y == 0) |> mutate(test = "Moved 1+ toward underdog -> underdog"),
  by_era(filter(m, toward_fav >= 2), y == 1) |> mutate(test = "Moved 2+ toward favorite -> favorite"),
  by_era(filter(m, toward_fav <= -2), y == 0) |> mutate(test = "Moved 2+ toward underdog -> underdog"),
  by_era(filter(m, toward_fav == 0), y == 0) |> mutate(test = "No move -> underdog")
) |> relocate(test) |> print(width = Inf)

cat("\n=== B. KEY NUMBERS ===\n")
crossed <- function(a, b, k) (a < k & b > k) | (a > k & b < k) | (a == k) != (b == k)
bind_rows(
  by_era(filter(m, crossed(open_fav, close_fav, 3) & toward_fav > 0), y == 1) |> mutate(test = "Crossed 3 toward favorite -> favorite"),
  by_era(filter(m, crossed(open_fav, close_fav, 3) & toward_fav < 0), y == 0) |> mutate(test = "Crossed 3 toward underdog -> underdog"),
  by_era(filter(m, crossed(open_fav, close_fav, 7) & toward_fav > 0), y == 1) |> mutate(test = "Crossed 7 toward favorite -> favorite"),
  by_era(filter(m, crossed(open_fav, close_fav, 7) & toward_fav < 0), y == 0) |> mutate(test = "Crossed 7 toward underdog -> underdog")
) |> relocate(test) |> print(width = Inf)

cat("\n=== C. CLOSING LINE VALUE OF OUR RULES (market moves toward our pick?) ===\n")
cat("CLV = points the line moved in favor of the side we picked, open to close.\n")
cat("Positive CLV means the market later agreed with us. Typical sharp bettors average ~+0.5.\n\n")
g <- m |> add_schedule_features()
scen <- read_rds(here("objects", "scenarios_recent.rds")); combo <- read_rds(here("objects", "combo_scenarios_two_era.rds"))
sets <- list(scenarios_23 = list(conds = scenario_conditions(g, scen$cuts), rules = scen$hits |> select(scenario, side)),
             combos_14 = list(conds = combo_conditions(g, combo$cuts), rules = combo$rules |> select(scenario, side = bet)),
             blowout_dog = list(conds = combo_conditions(g, combo$cuts), rules = tibble(scenario = "dog_blowout_loss", side = "underdog")))
clv <- imap_dfr(sets, function(s, set) map_dfr(seq_len(nrow(s$rules)), function(i) {
  hit <- scenario_fires(s$conds, s$rules$scenario[i])
  tibble(set, scenario = s$rules$scenario[i], side = s$rules$side[i], game_id_norm = g$game_id_norm[hit],
         # Bet the favorite at the open: a move toward the favorite means we laid fewer
         # points than the close -> positive CLV. For the underdog it's the reverse.
         season = g$season[hit], clv = if (s$rules$side[i] == "favorite") g$toward_fav[hit] else -g$toward_fav[hit],
         win = if (s$rules$side[i] == "favorite") g$y[hit] == 1 else g$y[hit] == 0)
}))
clv |> group_by(set) |>
  summarize(bets = n(), avg_clv = round(mean(clv), 2), pct_line_moved_our_way = round(100 * mean(clv > 0), 1),
            pct_moved_against = round(100 * mean(clv < 0), 1), win_pct = round(100 * mean(win), 1), .groups = "drop") |> print()
cat(sprintf("\nBaseline: avg CLV betting every underdog at the open = %+.2f, every favorite = %+.2f\n",
            -mean(m$toward_fav), mean(m$toward_fav)))
cat("\nBy rule (combos + blowout):\n")
clv |> filter(set != "scenarios_23") |> group_by(scenario, side) |>
  summarize(bets = n(), avg_clv = round(mean(clv), 2), moved_our_way = round(100 * mean(clv > 0)), win_pct = round(100 * mean(win), 1), .groups = "drop") |>
  arrange(desc(avg_clv)) |> print(n = Inf)

write_rds(m |> select(game_id_norm, season, week, open_home, close_home, toward_fav, src), here("objects", "line_movement.rds"))
