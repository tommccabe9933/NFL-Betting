# Luck vs point differential — pre-registered before testing
#
# Season-to-date, using only games already played (from week 4 on):
#   pythag  = PF^2.37 / (PF^2.37 + PA^2.37)   (expected win % from points)
#   luck    = actual win % - pythag            (positive = record flatters the team)
#   close_w = record in one-score games (margin <= 8), wins minus losses
# Rules (reason: the market prices records/narratives, points regress less):
#   R1 favorite luck >= +0.15                      -> underdog
#   R2 underdog luck <= -0.15                      -> underdog
#   R3 luck gap (fav - dog) in top 20% (2012-2017) -> underdog
#   R4 favorite won 2+ more one-score games than it lost, dog the reverse -> underdog
# Judged on 2012-2017 AND 2018-2025 separately, then 2026, plus closing line value.

library(tidyverse); library(here); library(nflreadr)

norm <- function(x) recode(x, LV = "LVR", LA = "LAR", OAK = "LVR", SD = "LAC", STL = "LAR")
s <- load_schedules(2012:2026) |> filter(game_type == "REG") |>
  mutate(game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", norm(away_team), "_", norm(home_team)))
tg <- bind_rows(
  s |> transmute(game_id_norm, season, week, gameday, team = norm(home_team), pf = home_score, pa = away_score),
  s |> transmute(game_id_norm, season, week, gameday, team = norm(away_team), pf = away_score, pa = home_score)) |>
  arrange(team, gameday) |>
  group_by(team, season) |>
  mutate(played = lag(cumsum(!is.na(pf)), default = 0),
         cpf = lag(cumsum(coalesce(pf, 0)), default = 0), cpa = lag(cumsum(coalesce(pa, 0)), default = 0),
         wins = lag(cumsum(coalesce(pf > pa, FALSE) + 0.5 * coalesce(pf == pa, FALSE)), default = 0),
         close_net = lag(cumsum(coalesce((abs(pf - pa) <= 8 & pf > pa) - (abs(pf - pa) <= 8 & pf < pa), 0L)), default = 0),
         pythag = cpf^2.37 / (cpf^2.37 + cpa^2.37),
         luck = if_else(played >= 3, wins / played - pythag, NA_real_)) |>
  ungroup() |> select(game_id_norm, team, luck, close_net, played)

v <- read_rds(here("objects", "modeling_data_v37.rds")) |>
  filter(season >= 2012) |>
  mutate(y = as.integer(fav_cover == "cover"),
         home = str_extract(game_id_norm, "[A-Z]+$"), away = str_match(game_id_norm, "_\\d+_([A-Z]+)_")[, 2],
         fav = if_else(fav_at_home == "yes", home, away), dog = if_else(fav_at_home == "yes", away, home)) |>
  left_join(tg |> rename_with(~ str_c("f_", .x), -c(game_id_norm, team)), by = c("game_id_norm", "fav" = "team")) |>
  left_join(tg |> rename_with(~ str_c("d_", .x), -c(game_id_norm, team)), by = c("game_id_norm", "dog" = "team")) |>
  mutate(luck_gap = f_luck - d_luck)

gap_cut <- quantile(v$luck_gap[v$season %in% 2012:2017], 0.8, na.rm = TRUE)
yes <- function(x) coalesce(x, FALSE)
rules <- list(
  R1_fav_lucky      = yes(v$f_luck >= 0.15),
  R2_dog_unlucky    = yes(v$d_luck <= -0.15),
  R3_luck_gap_top20 = yes(v$luck_gap >= gap_cut),
  R4_close_game_gap = yes(v$f_close_net >= 2 & v$d_close_net <= -2))

lm_ <- read_rds(here("objects", "line_movement.rds")) |> select(game_id_norm, toward_fav)
v <- v |> left_join(lm_, by = "game_id_norm")

rec <- function(w) if (length(w) == 0) "—" else sprintf("%d-%d (%.1f%%)", sum(w), sum(!w), 100 * mean(w))
out <- imap_dfr(rules, function(hit, name) {
  d <- v[hit & !is.na(v$fav_cover), ]; win <- d$y == 0   # every rule bets the underdog
  per <- d |> filter(season %in% 2018:2025) |> group_by(season) |> summarize(p = mean(y == 0))
  tibble(rule = name, per_season = round(sum(d$season %in% 2018:2025) / 8, 1),
         `2012-17` = rec(win[d$season <= 2017]), `2018-25` = rec(win[d$season %in% 2018:2025]),
         seasons_won = str_c(sum(per$p > 0.524), "/8"), `2026` = rec(win[d$season == 2026]),
         avg_clv = round(mean(-d$toward_fav, na.rm = TRUE), 2))
})
cat(sprintf("Luck-gap top-20%% cutoff (from 2012-2017): %.3f\n\n", gap_cut))
print(out, width = Inf)
cat("\nReference: every underdog —", rec(v$y[!is.na(v$fav_cover) & v$season %in% 2018:2025] == 0), "in 2018-25\n")
write_rds(list(gap_cut = gap_cut), here("objects", "luck_rule.rds"))
