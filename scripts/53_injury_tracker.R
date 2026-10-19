# Injury composite — 2026 paper-trading tracker
#
# THE RULE (frozen 2026-10-19, before any live pick was logged):
#   injury_net = mean of z(injury_diff), -z(fav_n_out), z(dog_n_out)
#     z-scores use 2012-2022 means/SDs (objects/injury_rule.rds)
#     positive = the underdog is more banged up than the favorite
#   Bet the FAVORITE when injury_net is in the top 20% of 2012-2022 games,
#   bet the UNDERDOG when it is in the bottom 20%, otherwise no bet.
#   Paper stake: 1 unit (risk 1.1 to win 1.0 at -110).
#
# Each run:
#   1. logs picks for upcoming games that haven't kicked off (a pick can be
#      updated until kickoff as injury reports change, then it is locked)
#   2. grades logged picks once final scores are in, at the logged line
#   3. prints the running record
# Weeks 1-6 of 2026 are backfilled and kept separate from live picks: their
# results were glimpsed (r = +0.13) when the composite was chosen.

library(tidyverse); library(here); library(nflreadr)

rule_path <- here("objects", "injury_rule.rds")
log_path  <- here("results", "injury_tracker_2026.csv")
v <- read_rds(here("objects", "modeling_data_v37.rds"))

# ---- 1. Freeze the rule (first run only) ----
if (!file.exists(rule_path)) {
  ref <- v |> filter(!is.na(fav_cover), season >= 2012, season <= 2022)
  stats <- map(set_names(c("injury_diff", "fav_n_out", "dog_n_out")),
               ~ c(mean = mean(ref[[.x]], na.rm = TRUE), sd = sd(ref[[.x]], na.rm = TRUE)))
  score_ref <- rowMeans(cbind((ref$injury_diff - stats$injury_diff[["mean"]]) / stats$injury_diff[["sd"]],
                              -(ref$fav_n_out - stats$fav_n_out[["mean"]]) / stats$fav_n_out[["sd"]],
                              (ref$dog_n_out - stats$dog_n_out[["mean"]]) / stats$dog_n_out[["sd"]]), na.rm = TRUE)
  write_rds(list(stats = stats,
                 fav_cutoff = unname(quantile(score_ref, 0.8, na.rm = TRUE)),
                 dog_cutoff = unname(quantile(score_ref, 0.2, na.rm = TRUE)),
                 frozen_on = as.character(Sys.Date())), rule_path)
}
rule <- read_rds(rule_path)
injury_net <- function(d) {
  s <- rule$stats
  rowMeans(cbind((d$injury_diff - s$injury_diff[["mean"]]) / s$injury_diff[["sd"]],
                 -(d$fav_n_out - s$fav_n_out[["mean"]]) / s$fav_n_out[["sd"]],
                 (d$dog_n_out - s$dog_n_out[["mean"]]) / s$dog_n_out[["sd"]]), na.rm = TRUE)
}

# ---- 2. Score games ----
sched <- load_schedules(2026) |> filter(game_type == "REG") |>
  mutate(home_team = recode(home_team, LV = "LVR", LA = "LAR"),
         away_team = recode(away_team, LV = "LVR", LA = "LAR"),
         game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", away_team, "_", home_team),
         kickoff = as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York"))

scored <- v |> filter(season == 2026) |>
  transmute(game_id_norm, week, line = spread_fav, injury_net = injury_net(pick(everything())),
            has_report = !is.na(fav_n_out) | !is.na(dog_n_out)) |>
  inner_join(sched |> select(game_id_norm, kickoff, home_team, away_team, spread_line, home_score, away_score),
             by = "game_id_norm") |>
  mutate(favorite = if_else(spread_line >= 0, home_team, away_team),
         underdog = if_else(spread_line >= 0, away_team, home_team),
         side = case_when(!has_report ~ "waiting for injury report",
                          injury_net >= rule$fav_cutoff ~ "favorite",
                          injury_net <= rule$dog_cutoff ~ "underdog", TRUE ~ "no bet"),
         pick = case_when(side == "favorite" ~ str_glue("{favorite} -{line}"),
                          side == "underdog" ~ str_glue("{underdog} +{line}"), TRUE ~ side) |> as.character())

old <- if (file.exists(log_path)) read_csv(log_path, show_col_types = FALSE,
                                          col_types = cols(.default = col_guess(), game_id_norm = "c")) else tibble(game_id_norm = character())
now <- Sys.time()

# Live picks: games not started yet (re-logged each run until kickoff)
live <- scored |> filter(kickoff > now) |>
  transmute(game_id_norm, week, kickoff = format(kickoff, "%Y-%m-%d %H:%M"), favorite, underdog, line,
            injury_net = round(injury_net, 3), side, pick, type = "live", logged_at = format(now, "%Y-%m-%d %H:%M"))
# Backfill: weeks already played that were never logged live
backfill <- scored |> filter(kickoff <= now, !(game_id_norm %in% old$game_id_norm)) |>
  transmute(game_id_norm, week, kickoff = format(kickoff, "%Y-%m-%d %H:%M"), favorite, underdog, line,
            injury_net = round(injury_net, 3), side, pick, type = "backfill", logged_at = format(now, "%Y-%m-%d %H:%M"))

log <- bind_rows(
  old |> filter(!(game_id_norm %in% live$game_id_norm)) |> select(-any_of(c("result", "units"))),
  live, backfill)

# ---- 3. Grade ----
graded <- log |>
  left_join(sched |> select(game_id_norm, home_team, home_score, away_score), by = "game_id_norm") |>
  mutate(fav_margin = if_else(favorite == home_team, home_score - away_score, away_score - home_score),
         beat = fav_margin - line,
         result = case_when(!side %in% c("favorite", "underdog") | is.na(home_score) ~ NA_character_,
                            beat == 0 ~ "push",
                            (side == "favorite") == (beat > 0) ~ "win", TRUE ~ "loss"),
         units = case_when(result == "win" ~ 1, result == "loss" ~ -1.1, result == "push" ~ 0)) |>
  select(-home_team, -home_score, -away_score, -fav_margin, -beat) |>
  arrange(week, kickoff)
write_csv(graded, log_path)

record <- function(d) d |> filter(!is.na(result)) |>
  summarize(bets = n(), wins = sum(result == "win"), losses = sum(result == "loss"),
            pushes = sum(result == "push"), win_pct = round(100 * wins / max(wins + losses, 1), 1),
            units = round(sum(units), 2))

cat(sprintf("Rule frozen %s | favorite if injury_net >= %.2f, underdog if <= %.2f\n\n",
            rule$frozen_on, rule$fav_cutoff, rule$dog_cutoff))
cat("RECORD (52.4% = breakeven)\n")
graded |> group_by(type) |> record() |> print()
cat("\nTHIS WEEK'S PAPER PICKS\n")
graded |> filter(type == "live", side != "no bet") |>
  select(week, kickoff, pick, injury_net) |> print(n = Inf)
n_wait <- sum(graded$type == "live" & graded$side == "waiting for injury report")
if (n_wait > 0) cat(sprintf("%d upcoming games are waiting for injury reports (out Wed-Fri) — re-run Saturday.\n", n_wait))
cat("\nLog:", log_path, "\n")
