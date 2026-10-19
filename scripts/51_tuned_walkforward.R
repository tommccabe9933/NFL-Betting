# Nested temporal tuning + week-by-week test
#
# For each test season S (2019-2026):
#   Inner (tuning) — only seasons before S:
#     fold 1: train seasons < S-2, validate on S-2
#     fold 2: train seasons < S-1, validate on S-1
#     pick the config with the lowest average validation log-loss, then the
#     betting edge (|p - 0.5| cutoff) with the best validation profit
#   Outer (test) — week by week through S, retraining on all games before
#     each week with the chosen config
# The test season never influences its own settings.

library(tidyverse); library(here); library(glmnet); library(xgboost)
source(here("scripts", "model_functions.R"))
set.seed(42)

version <- Sys.getenv("MODEL_VERSION", "v37")
v <- read_rds(here("objects", str_glue("modeling_data_{version}.rds"))) |>
  filter(!is.na(fav_cover)) |>
  mutate(y = as.integer(fav_cover == "cover"), t = season * 100 + week) |>
  mutate(
    pass_matchup    = (fav_off_pass_epa_roll5 - dog_def_pass_epa_roll5) - (dog_off_pass_epa_roll5 - fav_def_pass_epa_roll5),
    rush_matchup    = (fav_off_rush_epa_roll5 - dog_def_rush_epa_roll5) - (dog_off_rush_epa_roll5 - fav_def_rush_epa_roll5),
    success_matchup = (fav_off_success_rate_roll5 - dog_def_success_rate_roll5) - (dog_off_success_rate_roll5 - fav_def_success_rate_roll5),
    explosive_matchup = (fav_off_explosive_rate_roll5 - dog_def_explosive_rate_roll5) - (dog_off_explosive_rate_roll5 - fav_def_explosive_rate_roll5),
    form_delta = (fav_off_epa_play_roll3 - fav_off_epa_play_roll5) - (dog_off_epa_play_roll3 - dog_off_epa_play_roll5),
    out_delta  = fav_n_out - dog_n_out)

feature_sets <- list(
  deltas = c("pass_matchup", "rush_matchup", "success_matchup", "explosive_matchup", "form_delta",
             "qb_epa_per_play_roll5_diff", "injury_diff", "out_delta", "rest_diff", "elo_spread_residual", "spread_fav"),
  all = stable_features(v |> select(-y, -t))
)
start_season <- 2016

prep <- function(train, test, feats) {
  cov <- map_dbl(feats, ~ mean(!is.na(train[[.x]])))
  feats <- feats[cov > 0.5]
  med <- map_dbl(feats, ~ median(train[[.x]], na.rm = TRUE)) |> set_names(feats)
  to_m <- function(d) {
    m <- as.matrix(d[feats]); m[!is.finite(m)] <- NA
    for (f in feats) m[is.na(m[, f]), f] <- med[[f]]
    m
  }
  xtr <- to_m(train); xte <- to_m(test)
  mu <- colMeans(xtr); s <- apply(xtr, 2, sd); s[s == 0] <- 1
  list(xtr = scale(xtr, mu, s), xte = scale(xte, mu, s), ytr = train$y)
}

# Configs
lambdas <- 10^seq(-3, -0.5, length.out = 10)
enet_grid <- expand_grid(model = "enet", alpha = c(0, 0.25, 0.5, 0.75, 1))
xgb_grid <- tibble(model = "xgb",
                   max_depth = sample(1:4, 30, TRUE), eta = sample(c(0.01, 0.03, 0.1), 30, TRUE),
                   nrounds = sample(c(100, 300, 600), 30, TRUE), min_child_weight = sample(c(10, 30, 60), 30, TRUE),
                   subsample = sample(c(0.6, 0.8), 30, TRUE), colsample_bytree = sample(c(0.5, 0.8), 30, TRUE))

# Returns a matrix of predictions: one column per config (enet: per lambda)
predict_configs <- function(d, model, grid) {
  if (model == "enet") {
    map(seq_len(nrow(grid)), function(i) {
      fit <- glmnet(d$xtr, d$ytr, family = "binomial", alpha = grid$alpha[i], lambda = lambdas)
      p <- predict(fit, d$xte, s = lambdas, type = "response")
      colnames(p) <- str_glue("enet_a{grid$alpha[i]}_l{signif(lambdas, 2)}"); p
    }) |> do.call(what = cbind)
  } else {
    dtr <- xgb.DMatrix(d$xtr, label = d$ytr); dte <- xgb.DMatrix(d$xte)
    p <- map(seq_len(nrow(grid)), function(i) {
      g <- grid[i, ]
      m <- xgb.train(params = list(objective = "binary:logistic", max_depth = g$max_depth, eta = g$eta,
                                   min_child_weight = g$min_child_weight, subsample = g$subsample,
                                   colsample_bytree = g$colsample_bytree, lambda = 5, nthread = 4),
                     data = dtr, nrounds = g$nrounds, verbose = 0)
      predict(m, dte)
    }) |> do.call(what = cbind)
    colnames(p) <- str_glue("xgb_{seq_len(nrow(grid))}"); p
  }
}

logloss <- function(p, y) -mean(y * log(pmax(p, 1e-6)) + (1 - y) * log(pmax(1 - p, 1e-6)))
edges <- c(0.02, 0.04, 0.06, 0.08)
profit_at <- function(p, y, e) {
  bet <- case_when(p > 0.5 + e ~ 1L, p < 0.5 - e ~ 0L, TRUE ~ NA_integer_)
  win <- bet == y
  sum(if_else(win, 100, -110), na.rm = TRUE)
}

run_combo <- function(model, fs_name) {
  feats <- feature_sets[[fs_name]]
  grid <- if (model == "enet") enet_grid else xgb_grid
  map_dfr(2019:2026, function(S) {
    # --- inner tuning on S-2 and S-1 ---
    val <- map(c(S - 2, S - 1), function(vs) {
      tr <- v |> filter(season >= start_season, season < vs)
      va <- v |> filter(season == vs)
      list(p = predict_configs(prep(tr, va, feats), model, grid), y = va$y)
    })
    ll <- (apply(val[[1]]$p, 2, logloss, y = val[[1]]$y) + apply(val[[2]]$p, 2, logloss, y = val[[2]]$y)) / 2
    best <- which.min(ll)
    edge <- edges[which.max(map_dbl(edges, function(e)
      profit_at(val[[1]]$p[, best], val[[1]]$y, e) + profit_at(val[[2]]$p[, best], val[[2]]$y, e)))]
    cfg_name <- colnames(val[[1]]$p)[best]
    # Refit only the chosen config each week (enet columns are alpha x lambda)
    if (model == "enet") {
      chosen_row <- grid[ceiling(best / length(lambdas)), ]; chosen_col <- (best - 1) %% length(lambdas) + 1
    } else {
      chosen_row <- grid[best, ]; chosen_col <- 1
    }

    # --- outer: week by week through S with the chosen config ---
    weeks <- v |> filter(season == S) |> distinct(t) |> pull(t) |> sort()
    map_dfr(weeks, function(wt) {
      tr <- v |> filter(season >= start_season, t < wt); te <- v |> filter(t == wt)
      p <- predict_configs(prep(tr, te, feats), model, chosen_row)[, chosen_col]
      tibble(season = S, game_id_norm = te$game_id_norm, y = te$y, p = p)
    }) |>
      mutate(model = model, features = fs_name, config = cfg_name, edge = edge,
             bet = case_when(p > 0.5 + edge ~ 1L, p < 0.5 - edge ~ 0L, TRUE ~ NA_integer_))
  })
}

combos <- expand_grid(model = c("enet", "xgb"), fs = c("deltas", "all"))
preds <- pmap_dfr(combos, function(model, fs) { cat("Running", model, fs, "\n"); run_combo(model, fs) })
write_rds(preds, here("objects", str_glue("tuned_walkforward_{version}.rds")))

bets <- preds |> filter(!is.na(bet)) |> mutate(win = bet == y)
summ <- function(d) d |> summarize(bets = n(), win_pct = round(100 * mean(win), 1),
                                   profit = sum(if_else(win, 100, -110)), .groups = "drop")

cat("\n=== NESTED-TUNED, WEEK-BY-WEEK — win % by season (52.4% = breakeven) ===\n")
bets |> group_by(model, features, season) |> summ() |>
  select(model, features, season, win_pct) |>
  pivot_wider(names_from = season, values_from = win_pct) |> print(width = Inf)

cat("\nTotals:\n")
bets |> mutate(period = if_else(season == 2026, "2026", "2019_25")) |>
  group_by(model, features, period) |> summ() |>
  pivot_wider(names_from = period, values_from = c(bets, win_pct, profit)) |> print(width = Inf)

cat("\nLog-loss vs coin flip (0.693) on all test games, 2019-2025:\n")
preds |> filter(season <= 2025) |> group_by(model, features) |>
  summarize(logloss = round(logloss(p, y), 4), .groups = "drop") |> print()

cat("\nChosen settings per season:\n")
preds |> distinct(model, features, season, config, edge) |> arrange(model, features, season) |> print(n = Inf)
write_csv(bets |> group_by(model, features, season) |> summ(), here("results", str_glue("tuned_walkforward_{version}.csv")))
