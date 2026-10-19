# Scenario paper tracker — 2026
#
# Tracks the scenarios locked in objects/scenarios_recent.rds (found on
# 2018-2025 by 54_scenario_search.R, MODE=recent; cutoffs frozen with them).
# Caveat recorded at lock time: shuffled outcomes produce about as many
# passing scenarios (~25), so these may be luck. 2026 is the test.
#
# Each run logs, per game, which scenarios fire and which side they back;
# picks update until kickoff and then lock. Records reported:
#   - each scenario on its own
#   - portfolio: one bet per game on the side most scenarios back (ties skipped)
#   - baseline: always bet the underdog, every game
#   - baseline on the portfolio's games: favorite/underdog base rate there
# Weeks already played when tracking started are marked "backfill".

library(tidyverse); library(here); library(nflreadr)
source(here("scripts", "scenarios.R"))

locked <- read_rds(here("objects", "scenarios_recent.rds"))
rules <- locked$hits |> select(scenario, side)
log_path <- here("results", "scenario_tracker_2026.csv")

v <- read_rds(here("objects", "modeling_data_v37.rds")) |> filter(season == 2026)
conds <- scenario_conditions(v, locked$cuts)

fired <- map_dfr(seq_len(nrow(rules)), function(i) {
  hit <- scenario_fires(conds, rules$scenario[i])
  tibble(game_id_norm = v$game_id_norm[hit], scenario = rules$scenario[i], side = rules$side[i])
})

sched <- load_schedules(2026) |> filter(game_type == "REG") |>
  mutate(home_team = recode(home_team, LV = "LVR", LA = "LAR"),
         away_team = recode(away_team, LV = "LVR", LA = "LAR"),
         game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", away_team, "_", home_team),
         kickoff = as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York"))

games <- v |> select(game_id_norm, week, line = spread_fav) |>
  inner_join(sched |> select(game_id_norm, kickoff, home_team, away_team, spread_line), by = "game_id_norm") |>
  left_join(fired |> group_by(game_id_norm) |>
              summarize(scenarios = paste(str_c(scenario, " [", side, "]"), collapse = "; "),
                        fav_votes = sum(side == "favorite"), dog_votes = sum(side == "underdog")),
            by = "game_id_norm") |>
  mutate(across(c(fav_votes, dog_votes), ~ coalesce(.x, 0L)),
         favorite = if_else(spread_line >= 0, home_team, away_team),
         underdog = if_else(spread_line >= 0, away_team, home_team),
         portfolio_side = case_when(fav_votes > dog_votes ~ "favorite", dog_votes > fav_votes ~ "underdog",
                                    TRUE ~ "no bet"),
         pick = case_when(portfolio_side == "favorite" ~ str_glue("{favorite} -{line}"),
                          portfolio_side == "underdog" ~ str_glue("{underdog} +{line}"), TRUE ~ "no bet") |> as.character())

old <- if (file.exists(log_path)) read_csv(log_path, show_col_types = FALSE, col_types = cols(.default = col_guess(), game_id_norm = "c")) else tibble(game_id_norm = character())
now <- Sys.time()
fmt <- function(d, type) d |> transmute(game_id_norm, week, kickoff = format(kickoff, "%Y-%m-%d %H:%M"), favorite, underdog,
                                        line, scenarios, fav_votes, dog_votes, portfolio_side, pick, type,
                                        logged_at = format(now, "%Y-%m-%d %H:%M"))
live <- fmt(games |> filter(kickoff > now), "live")
backfill <- fmt(games |> filter(kickoff <= now, !(game_id_norm %in% old$game_id_norm)), "backfill")
log <- bind_rows(old |> filter(!(game_id_norm %in% live$game_id_norm)) |> select(-any_of(c("fav_beat", "fav_result"))),
                 live, backfill)

# Grade from the favorite's side at the logged line
log <- log |>
  left_join(sched |> select(game_id_norm, home_team, home_score, away_score), by = "game_id_norm") |>
  mutate(fav_margin = if_else(favorite == home_team, home_score - away_score, away_score - home_score),
         fav_beat = fav_margin - line,
         fav_result = case_when(is.na(fav_beat) ~ NA_character_, fav_beat > 0 ~ "fav covered",
                                fav_beat < 0 ~ "dog covered", TRUE ~ "push")) |>
  select(-home_team, -home_score, -away_score, -fav_margin) |>
  arrange(week, kickoff)
write_csv(log, log_path)

graded <- log |> filter(!is.na(fav_result), fav_result != "push")
rec <- function(win) tibble(bets = length(win), wins = sum(win), win_pct = round(100 * mean(win), 1),
                            units = round(sum(if_else(win, 1, -1.1)), 1))

port <- graded |> filter(portfolio_side != "no bet")
summary_tbl <- bind_rows(
  rec((port$portfolio_side == "favorite") == (port$fav_result == "fav covered")) |> mutate(strategy = "Scenario portfolio"),
  rec(graded$fav_result == "dog covered") |> mutate(strategy = "Always underdog (all games)"),
  rec(port$fav_result == "dog covered") |> mutate(strategy = "Always underdog (portfolio's games)")
) |> relocate(strategy)

per_scenario <- graded |>
  separate_rows(scenarios, sep = "; ") |> filter(!is.na(scenarios)) |>
  mutate(side = str_extract(scenarios, "(?<=\\[)\\w+"), scenario = str_remove(scenarios, " \\[.*"),
         win = (side == "favorite") == (fav_result == "fav covered")) |>
  group_by(scenario, side) |>
  summarize(bets = n(), wins = sum(win), win_pct = round(100 * mean(win), 1), .groups = "drop") |>
  arrange(desc(bets))

cat(sprintf("Scenarios locked %s (%d rules, found on 2018-2025)\n\n", locked$frozen_on, nrow(rules)))
cat("2026 RECORD — all graded games so far (52.4% = breakeven)\n")
print(summary_tbl, width = Inf)
cat("\nBy scenario:\n"); print(per_scenario, n = Inf)
cat("\nUPCOMING PAPER PICKS (portfolio)\n")
log |> filter(type == "live", portfolio_side != "no bet") |>
  select(week, kickoff, pick, fav_votes, dog_votes) |> print(n = Inf)
cat("\nLog:", log_path, "\n")
