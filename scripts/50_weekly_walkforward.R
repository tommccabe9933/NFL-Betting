# Week-by-week walk-forward test
#
# Strictest temporal test: for every week from 2019 on, train only on games
# from earlier weeks, predict that week, then move forward. Nothing from the
# week being predicted (or later) is ever used. 2026 weeks are reported
# separately — no earlier analysis looked at them.
#
# Strategies:
#   always_fav / always_dog  — baselines
#   base_rate_follow         — bet whichever side covered more over the last 100 games
#   delta_model              — logistic regression on team-vs-team matchup deltas
#   delta_model_recent       — same, trained only on the last 3 seasons (adapts to regimes)
#   delta_enet               — elastic net (alpha 0.5), lambda tuned by season-blocked CV
#   delta_xgb                — shallow XGBoost (depth 2, heavy regularization)

library(tidyverse); library(here); library(glmnet); library(xgboost)
set.seed(42)

version <- Sys.getenv("MODEL_VERSION", "v37")
v <- read_rds(here("objects", str_glue("modeling_data_{version}.rds"))) |>
  filter(!is.na(fav_cover)) |>
  mutate(y = as.integer(fav_cover == "cover"), t = season * 100 + week)

# Matchup deltas: each side's offense against the other side's defense
# (defense EPA: lower = better, so offense - opposing defense = expected edge)
v <- v |> mutate(
  pass_matchup    = (fav_off_pass_epa_roll5 - dog_def_pass_epa_roll5) - (dog_off_pass_epa_roll5 - fav_def_pass_epa_roll5),
  rush_matchup    = (fav_off_rush_epa_roll5 - dog_def_rush_epa_roll5) - (dog_off_rush_epa_roll5 - fav_def_rush_epa_roll5),
  success_matchup = (fav_off_success_rate_roll5 - dog_def_success_rate_roll5) - (dog_off_success_rate_roll5 - fav_def_success_rate_roll5),
  explosive_matchup = (fav_off_explosive_rate_roll5 - dog_def_explosive_rate_roll5) - (dog_off_explosive_rate_roll5 - fav_def_explosive_rate_roll5),
  form_delta      = (fav_off_epa_play_roll3 - fav_off_epa_play_roll5) - (dog_off_epa_play_roll3 - dog_off_epa_play_roll5),
  qb_delta        = qb_epa_per_play_roll5_diff,
  injury_delta    = injury_diff,
  out_delta       = fav_n_out - dog_n_out,
  rest_delta      = rest_diff,
  market_gap      = elo_spread_residual
)
deltas <- c("pass_matchup", "rush_matchup", "success_matchup", "explosive_matchup", "form_delta",
            "qb_delta", "injury_delta", "out_delta", "rest_delta", "market_gap", "spread_fav")

# Median-fill and standardize with training-window stats only
prep_xy <- function(train, test) {
  med <- map_dbl(deltas, ~ median(train[[.x]], na.rm = TRUE)) |> set_names(deltas)
  fill <- function(d) d |> mutate(across(all_of(deltas), ~ replace_na(.x, med[cur_column()])))
  tr <- fill(train); te <- fill(test)
  mu <- map_dbl(deltas, ~ mean(tr[[.x]])); s <- map_dbl(deltas, ~ sd(tr[[.x]]))
  list(xtr = scale(as.matrix(tr[deltas]), mu, s), xte = scale(as.matrix(te[deltas]), mu, s),
       ytr = tr$y, season = tr$season)
}

fit_delta <- function(train, test) {
  d <- prep_xy(train, test)
  m <- glm(y ~ ., family = binomial, data = bind_cols(as_tibble(d$xtr), y = d$ytr))
  predict(m, as_tibble(d$xte), type = "response")
}

fit_enet <- function(train, test) {
  d <- prep_xy(train, test)
  cv <- cv.glmnet(d$xtr, d$ytr, family = "binomial", alpha = 0.5,
                  foldid = as.integer(factor(d$season)))
  as.vector(predict(cv, d$xte, s = "lambda.min", type = "response"))
}

fit_xgb <- function(train, test) {
  d <- prep_xy(train, test)
  m <- xgb.train(params = list(objective = "binary:logistic", max_depth = 2, eta = 0.03,
                               subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 30,
                               lambda = 5, nthread = 4),
                 data = xgb.DMatrix(d$xtr, label = d$ytr), nrounds = 150, verbose = 0)
  predict(m, xgb.DMatrix(d$xte))
}

side <- function(p, edge = 0.03) case_when(p > 0.5 + edge ~ "fav", p < 0.5 - edge ~ "dog", TRUE ~ "pass")

weeks <- v |> filter(season >= 2019) |> distinct(season, week, t) |> arrange(t)
bets <- map_dfr(seq_len(nrow(weeks)), function(i) {
  wk <- weeks[i, ]
  past <- v |> filter(t < wk$t, season >= 2012)
  te <- v |> filter(t == wk$t)
  last100 <- past |> slice_max(t, n = 100, with_ties = FALSE)
  p_all <- fit_delta(past |> filter(season >= 2016), te)
  p_rec <- fit_delta(past |> filter(season >= wk$season - 3), te)
  p_enet <- fit_enet(past |> filter(season >= 2016), te)
  p_xgb <- fit_xgb(past |> filter(season >= 2016), te)
  tibble(season = wk$season, week = wk$week, game_id_norm = te$game_id_norm, y = te$y,
         always_fav = "fav", always_dog = "dog",
         base_rate_follow = if_else(mean(last100$y) >= 0.5, "fav", "dog"),
         delta_model = side(p_all), delta_model_recent = side(p_rec),
         delta_enet = side(p_enet), delta_xgb = side(p_xgb))
}) |>
  pivot_longer(always_fav:delta_xgb, names_to = "strategy", values_to = "bet") |>
  filter(bet != "pass") |>
  mutate(win = (bet == "fav" & y == 1) | (bet == "dog" & y == 0))

summ <- function(d) d |> summarize(bets = n(), win_pct = round(100 * mean(win), 1),
                                   profit = sum(if_else(win, 100, -110)), .groups = "drop")

cat("=== WEEK-BY-WEEK WALK-FORWARD (train only on earlier weeks) ===\n")
cat("Win % by season (52.4% = breakeven at -110)\n\n")
by_season <- bets |> group_by(strategy, season) |> summ()
by_season |> select(strategy, season, win_pct) |>
  pivot_wider(names_from = season, values_from = win_pct) |> print(width = Inf)

cat("\nTotals 2019-2025 vs fresh 2026 (weeks 1-6, never examined before):\n")
bets |> mutate(period = if_else(season == 2026, "2026 (fresh)", "2019-2025")) |>
  group_by(strategy, period) |> summ() |>
  pivot_wider(names_from = period, values_from = c(bets, win_pct, profit)) |> print(width = Inf)

write_csv(by_season, here("results", str_glue("weekly_walkforward_{version}.csv")))
