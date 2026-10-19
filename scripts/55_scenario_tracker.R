# Scenario paper tracker — 2026
#
# Rule sets (all locked before the games they're graded on):
#   scenarios_23 — objects/scenarios_recent.rds (54, MODE=recent; found on
#                  2018-2025, about as many pass with shuffled outcomes)
#   combos_14    — objects/combo_scenarios_two_era.rds (57; found on
#                  2018-2025 AND >= 55% in untouched 2012-2017)
#   blowout_dog  — underdog off a 21+ point loss (56; reason written first)
#
# Each run logs which rules fire on each game (picks update until kickoff,
# then lock), grades finished games at the logged line, and reports each
# set's portfolio (one bet per game, majority side) next to baselines.
# Weeks already played when a set was added are marked "backfill".

library(tidyverse); library(here); library(nflreadr)
source(here("scripts", "scenarios.R"))
source(here("scripts", "schedule_features.R"))

log_path <- here("results", "scenario_tracker_2026.csv")
v <- read_rds(here("objects", "modeling_data_v37.rds")) |> filter(season == 2026)
g <- add_schedule_features(v)

scen <- read_rds(here("objects", "scenarios_recent.rds"))
combo <- read_rds(here("objects", "combo_scenarios_two_era.rds"))
combo_conds <- combo_conditions(g, combo$cuts)

rule_sets <- list(
  scenarios_23 = list(conds = scenario_conditions(v, scen$cuts), rules = scen$hits |> select(scenario, side)),
  combos_14    = list(conds = combo_conds, rules = combo$rules |> select(scenario, side = bet)),
  blowout_dog  = list(conds = combo_conds, rules = tibble(scenario = "dog_blowout_loss", side = "underdog")))

fired <- imap_dfr(rule_sets, function(rs, set) map_dfr(seq_len(nrow(rs$rules)), function(i) {
  hit <- scenario_fires(rs$conds, rs$rules$scenario[i])
  tibble(game_id_norm = v$game_id_norm[hit], set = set, scenario = rs$rules$scenario[i], side = rs$rules$side[i])
}))

sched <- load_schedules(2026) |> filter(game_type == "REG") |>
  mutate(home_team = recode(home_team, LV = "LVR", LA = "LAR"),
         away_team = recode(away_team, LV = "LVR", LA = "LAR"),
         game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", away_team, "_", home_team),
         kickoff = as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York"))

games <- v |> select(game_id_norm, week, line = spread_fav) |>
  inner_join(sched |> select(game_id_norm, kickoff, home_team, away_team, spread_line), by = "game_id_norm") |>
  mutate(favorite = if_else(spread_line >= 0, home_team, away_team),
         underdog = if_else(spread_line >= 0, away_team, home_team))

now <- Sys.time()
rows <- games |>
  left_join(fired, by = "game_id_norm", relationship = "one-to-many") |>
  transmute(game_id_norm, week, kickoff_dt = kickoff, kickoff = format(kickoff, "%Y-%m-%d %H:%M"),
            favorite, underdog, line, set, scenario, side, logged_at = format(now, "%Y-%m-%d %H:%M"))

old <- if (file.exists(log_path)) read_csv(log_path, show_col_types = FALSE,
  col_types = cols(.default = col_guess(), game_id_norm = "c", set = "c", scenario = "c", side = "c")) else
  tibble(game_id_norm = character(), set = character())
old <- old |> filter("set" %in% names(old))  # older log format is replaced

live <- rows |> filter(kickoff_dt > now) |> mutate(type = "live")
# Backfill games already played that this set never logged
logged_keys <- old |> filter(!is.na(set)) |> distinct(game_id_norm, set)
backfill <- rows |> filter(kickoff_dt <= now) |>
  anti_join(logged_keys, by = c("game_id_norm", "set")) |> mutate(type = "backfill")
log <- bind_rows(old |> filter(!(game_id_norm %in% live$game_id_norm)) |> select(-any_of(c("fav_beat", "fav_result"))),
                 live |> select(-kickoff_dt), backfill |> select(-kickoff_dt)) |>
  filter(!is.na(set) | !duplicated(game_id_norm))

log <- log |>
  left_join(sched |> select(game_id_norm, home_team, home_score, away_score), by = "game_id_norm") |>
  mutate(fav_beat = if_else(favorite == home_team, home_score - away_score, away_score - home_score) - line,
         fav_result = case_when(is.na(fav_beat) ~ NA_character_, fav_beat > 0 ~ "fav covered",
                                fav_beat < 0 ~ "dog covered", TRUE ~ "push")) |>
  select(-home_team, -home_score, -away_score) |>
  arrange(week, kickoff, set)
write_csv(log, log_path)

# ---- Report ----
rec <- function(win) tibble(bets = length(win), wins = sum(win), win_pct = round(100 * mean(win), 1),
                            units = round(sum(if_else(win, 1, -1.1)), 1))
graded <- log |> filter(!is.na(fav_result), fav_result != "push")
portfolio <- function(d) d |> filter(!is.na(set)) |>
  group_by(set, game_id_norm, fav_result) |>
  summarize(fav_votes = sum(side == "favorite"), dog_votes = sum(side == "underdog"), .groups = "drop") |>
  filter(fav_votes != dog_votes) |>
  mutate(bet = if_else(fav_votes > dog_votes, "favorite", "underdog"))

port <- portfolio(graded)
all_games <- graded |> distinct(game_id_norm, fav_result)
summary_tbl <- bind_rows(
  port |> group_by(strategy = set) |>
    reframe(rec((bet == "favorite") == (fav_result == "fav covered"))),
  port |> group_by(strategy = str_c(set, " — underdog baseline")) |>
    reframe(rec(fav_result == "dog covered")),
  rec(all_games$fav_result == "dog covered") |> mutate(strategy = "Always underdog (every game)"))

per_rule <- graded |> filter(!is.na(set)) |>
  mutate(win = (side == "favorite") == (fav_result == "fav covered")) |>
  group_by(set, scenario, side) |>
  summarize(bets = n(), wins = sum(win), win_pct = round(100 * mean(win), 1), .groups = "drop") |>
  arrange(set, desc(bets))

cat("2026 RECORD — portfolios vs underdog baselines (52.4% = breakeven)\n")
print(summary_tbl |> arrange(strategy), width = Inf)
cat("\nBy rule:\n"); print(per_rule, n = Inf)

cat("\nUPCOMING PAPER PICKS\n")
upcoming <- portfolio(log |> filter(type == "live") |> mutate(fav_result = NA_character_)) |>
  left_join(games |> select(game_id_norm, week, kickoff, favorite, underdog, line), by = "game_id_norm") |>
  mutate(pick = if_else(bet == "favorite", str_glue("{favorite} -{line}"), str_glue("{underdog} +{line}"))) |>
  arrange(kickoff)
upcoming |> transmute(week, kickoff = format(kickoff, "%a %m-%d %H:%M"), set, pick, votes = str_c(fav_votes, " fav / ", dog_votes, " dog")) |>
  print(n = Inf)
cat("\nLog:", log_path, "\n")
