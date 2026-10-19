# Neutralized signal scan
#
# Raw correlations can be borrowed from other variables (especially the
# spread). Here each feature is tested with everything else held constant:
#   controls = market (spread, total, Elo, home, week)
#            + top 20 principal components of all OTHER features
#              (excluding the tested feature and anything |r| > 0.5 with it)
# Both the feature and the cover outcome are residualized on the controls;
# the partial correlation is the correlation of the two residuals.
# A shuffled-outcome run gives the strongest partial r expected from luck.

library(tidyverse); library(here)
set.seed(42)

version <- Sys.getenv("MODEL_VERSION", "v37")
v <- read_rds(here("objects", str_glue("modeling_data_{version}.rds"))) |>
  filter(!is.na(fav_cover)) |>
  mutate(y = as.integer(fav_cover == "cover"), fav_home = as.integer(fav_at_home == "yes"))

market <- c("spread_fav", "over_under_line", "elo_diff", "fav_home", "week")
feats <- v |> select(where(is.numeric), -season, -y, -any_of(market)) |> names()

prep <- function(d) {
  m <- d |> select(all_of(c(market, feats))) |> as.matrix()
  m[!is.finite(m)] <- NA
  med <- apply(m, 2, median, na.rm = TRUE)
  for (j in seq_len(ncol(m))) m[is.na(m[, j]), j] <- med[j]
  keep <- apply(m, 2, sd) > 0
  m[, keep, drop = FALSE]
}

# Returns partial r for the real outcome plus `n_null` shuffled outcomes.
# Controls are built once per feature and reused for every shuffle.
neutralized_scan <- function(d, n_pc = 20, n_null = 0) {
  m <- prep(d)
  fs <- intersect(feats, colnames(m))
  cm <- cor(m[, fs])
  ys <- cbind(d$y, replicate(n_null, sample(d$y)))
  map_dfr(fs, function(f) {
    others <- fs[!is.na(cm[f, ]) & abs(cm[f, ]) <= 0.5]
    zs <- scale(m[, others])
    pcs <- zs %*% eigen(cor(zs), symmetric = TRUE)$vectors[, 1:n_pc]
    q <- qr(cbind(1, m[, intersect(market, colnames(m))], pcs))
    rx <- qr.resid(q, m[, f]); ry <- qr.resid(q, ys)
    r <- as.vector(cor(rx, ry))
    tibble(feature = f, partial_r = r[1], raw_r = cor(m[, f], d$y),
           null_r = list(r[-1]))
  })
}

recent <- v |> filter(season >= 2023)
past   <- v |> filter(season >= 2016, season <= 2022)

res_recent <- neutralized_scan(recent, n_null = 100)
res_past   <- neutralized_scan(past) |> select(feature, partial_r_2016_22 = partial_r)

# Luck baseline: strongest |partial r| across all features in each shuffle
null_max <- res_recent$null_r |> do.call(what = rbind) |> abs() |> apply(2, max)

per_season <- map_dfr(2023:2025, function(s) {
  neutralized_scan(v |> filter(season == s)) |> transmute(feature, season = s, partial_r)
}) |> pivot_wider(names_from = season, values_from = partial_r, names_prefix = "r_")

out <- res_recent |>
  select(-null_r) |>
  left_join(res_past, by = "feature") |>
  left_join(per_season, by = "feature") |>
  mutate(same_sign_2023_25 = sign(r_2023) == sign(r_2024) & sign(r_2024) == sign(r_2025),
         held_2016_22 = sign(partial_r_2016_22) == sign(partial_r)) |>
  arrange(desc(abs(partial_r)))

cat(sprintf("2023-25: %d games | %d features tested\n", nrow(recent), nrow(out)))
cat(sprintf("Strongest |partial r| from luck (100 shuffles): median %.3f, 95th pct %.3f\n\n",
  median(null_max), quantile(null_max, 0.95)))
print(out |> head(20) |> mutate(across(where(is.numeric), ~ round(.x, 3))), n = 20, width = 200)
write_csv(out, here("results", str_glue("neutralized_signals_{version}.csv")))
