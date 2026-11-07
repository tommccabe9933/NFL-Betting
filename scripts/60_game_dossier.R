# Game dossier — every historically tested data point for each upcoming game
#
# For each game not yet kicked off:
#   - rules that fire (combos_14 with both-era records, scenarios_23, blowout_dog)
#   - each starting QB's career ATS in his exact role (fav/dog x home/road), 2012+
#   - season-to-date efficiency (net EPA/play) ranks and the spread they imply
#   - rest / bye edge
# Output: results/dossier_<season>_wk<week>.csv and a printed summary.

library(tidyverse); library(here); library(nflreadr)

norm <- function(x) recode(x, LV = "LVR", LA = "LAR", OAK = "LVR", SD = "LAC", STL = "LAR")
season_now <- 2026
s_all <- load_schedules(2012:season_now) |> filter(game_type == "REG") |>
  mutate(game_id_norm = str_c(season, "_", str_pad(week, 2, pad = "0"), "_", norm(away_team), "_", norm(home_team)),
         kickoff = as.POSIXct(str_c(gameday, " ", gametime), tz = "America/New_York"))
upcoming <- s_all |> filter(season == season_now, is.na(result), kickoff > Sys.time(), !is.na(spread_line)) |>
  filter(week == min(week))

# ---- QB role records (career, 2012+) ----
tg <- bind_rows(
  s_all |> transmute(season, team = norm(home_team), qb = home_qb_name, home = TRUE, margin = result, fav_by = spread_line),
  s_all |> transmute(season, team = norm(away_team), qb = away_qb_name, home = FALSE, margin = -result, fav_by = -spread_line)) |>
  filter(!is.na(margin), !is.na(fav_by)) |>
  mutate(ats = margin - fav_by, role = if_else(fav_by > 0, "favorite", "underdog")) |> filter(ats != 0)
qb_role <- function(q, is_home, role_) {
  d <- tg |> filter(qb == q, role == role_)
  dh <- d |> filter(home == is_home)
  fmt <- function(x) if (nrow(x) < 8) NA_character_ else sprintf("%d-%d (%.0f%%)", sum(x$ats > 0), sum(x$ats < 0), 100 * mean(x$ats > 0))
  c(role = fmt(d), role_site = fmt(dh), pct_role = if (nrow(d) >= 8) mean(d$ats > 0) else NA, n_role = nrow(d))
}

# ---- Season efficiency ----
p <- load_pbp(season_now) |> filter(season_type == "REG", !is.na(epa), pass == 1 | rush == 1)
eff <- p |> group_by(team = norm(posteam)) |> summarize(off = mean(epa)) |>
  inner_join(p |> group_by(team = norm(defteam)) |> summarize(def = mean(epa)), by = "team") |>
  mutate(net = off - def, net_rank = rank(-net), off_rank = rank(-off), def_rank = rank(def))
# Points per unit of net-EPA gap, fit on 2018-2025 season-to-date data is overkill here;
# a plain league rule of thumb (~ 60 pts per 1.0 EPA/play gap over ~60 plays) is used,
# plus 1.5 for home field. Descriptive only — not a tested predictor.
implied <- function(h, a) round(60 * (eff$net[eff$team == h] - eff$net[eff$team == a]) + 1.5, 1)

# ---- Rules that fire ----
log <- read_csv(here("results", "scenario_tracker_2026.csv"), show_col_types = FALSE,
                col_types = cols(.default = col_guess(), kickoff = "c", logged_at = "c"))
combo <- read_rds(here("objects", "combo_scenarios_two_era.rds"))$rules |>
  transmute(scenario, side = bet, rec = sprintf("%.0f%%/%.0f%%", 100 * win_2018_25, 100 * win_2012_17))
scen <- read_rds(here("objects", "scenarios_recent.rds"))$hits |>
  transmute(scenario, side, rec = sprintf("%.0f%% (%d/8 yrs)", 100 * disc_pct, seasons_ok))
rules <- log |> filter(!is.na(set)) |>
  left_join(bind_rows(combo |> mutate(set = "combos_14"), scen |> mutate(set = "scenarios_23"),
                      tibble(set = "blowout_dog", scenario = "dog_blowout_loss", side = "underdog", rec = "56%")),
            by = c("set", "scenario", "side"))

out <- upcoming |> mutate(home = norm(home_team), away = norm(away_team),
                          fav = if_else(spread_line >= 0, home, away), dog = if_else(spread_line >= 0, away, home),
                          line = abs(spread_line)) |>
  rowwise() |>
  mutate(
    fav_qb = if_else(fav == home, home_qb_name, away_qb_name), dog_qb = if_else(fav == home, away_qb_name, home_qb_name),
    fav_qb_rec = qb_role(fav_qb, fav == home, "favorite")[["role_site"]],
    dog_qb_rec = qb_role(dog_qb, dog == home, "underdog")[["role_site"]],
    fav_rank = eff$net_rank[eff$team == fav], dog_rank = eff$net_rank[eff$team == dog],
    eff_implied_home_spread = implied(home, away),
    rest_edge = if_else(fav == home, home_rest - away_rest, away_rest - home_rest),
    fav_votes = sum(rules$side[rules$game_id_norm == game_id_norm] == "favorite"),
    dog_votes = sum(rules$side[rules$game_id_norm == game_id_norm] == "underdog"),
    combos = paste(unique(rules$rec[rules$game_id_norm == game_id_norm & rules$set == "combos_14"]), collapse = "; ")) |>
  ungroup() |>
  transmute(game_id_norm, kickoff = format(kickoff, "%a %H:%M"), fav, line, dog, total_line,
            fav_qb, fav_qb_rec, dog_qb, dog_qb_rec, fav_rank, dog_rank, eff_implied_home_spread,
            rest_edge, fav_votes, dog_votes, combos)

write_csv(out, here("results", str_glue("dossier_{season_now}_wk{str_pad(max(upcoming$week), 2, pad = '0')}.csv")))
print(out, n = Inf, width = Inf)
cat("\nRules detail:\n")
rules |> filter(game_id_norm %in% out$game_id_norm) |>
  transmute(game_id_norm, set, scenario, side, rec) |> arrange(game_id_norm, set) |> print(n = Inf, width = Inf)
