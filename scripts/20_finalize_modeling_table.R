# Finalize the v3.6 modeling table
# 19c output + composites (frozen scaling) → same 300 columns as v3_FINAL,
# now with complete 2025 rows and 2026 rows (including upcoming games).
#
# Run order: 02 → 01 → 10 → Python pipeline → 19c → this script.

library(tidyverse); library(here)
source(here("scripts", "composites.R"))

v3_final <- read_rds(here("objects", "modeling_data_v3_FINAL.rds"))

# v36: composite scaling frozen at the original 2012-2025 values, for an
#      exact match with v3_FINAL (built on the swapped fav/dog columns).
# v37: fav/dog fixed upstream (src/normalize/games.py); scaling learned on
#      2012-2022 only, so no test season (2023+) feeds the composites.
version <- Sys.getenv("MODEL_VERSION", "v37")
base <- read_rds(here("objects", "modeling_data_v3.rds"))

if (version == "v36") {
  scaling_path <- here("objects", "composite_scaling.rds")
  if (!file.exists(scaling_path)) {
    write_rds(attr(add_all_composites(v3_final), "composite_scaling"), scaling_path)
  }
  scaling <- read_rds(scaling_path)
} else {
  scaling <- attr(add_all_composites(base |> filter(season <= 2022)), "composite_scaling")
  write_rds(scaling, here("objects", str_glue("composite_scaling_{version}.rds")))
}

v36 <- base |>
  add_all_composites(scaling) |>
  select(all_of(names(v3_final)))
attr(v36, "composite_scaling") <- NULL

upcoming <- read_rds(here("objects", "upcoming_games.rds"))
v36 <- v36 |> mutate(is_upcoming = game_id_norm %in% upcoming$game_id_norm)

# Spot check: rows that existed before should mostly be unchanged
old <- v3_final |> filter(season <= 2024)
chk <- v36 |> semi_join(old, by = "game_id_norm") |> arrange(game_id_norm)
old <- old |> semi_join(chk, by = "game_id_norm") |> arrange(game_id_norm)
num_cols <- names(old)[map_lgl(old, is.numeric)]
agree <- map_dbl(num_cols, ~ mean(abs(old[[.x]] - chk[[.x]]) < 1e-6 | (is.na(old[[.x]]) & is.na(chk[[.x]]))))
cat(sprintf("2012-2024: %d of %d games rebuilt | columns ≥99%% identical: %d of %d\n",
  nrow(chk), nrow(v3_final |> filter(season <= 2024)), sum(agree >= 0.99), length(num_cols)))
if (any(agree < 0.99)) {
  cat("Columns that changed:\n")
  print(tibble(column = num_cols, agree = round(agree, 3)) |> filter(agree < 0.99) |> arrange(agree), n = 50)
}

cov <- v36 |> filter(season >= 2024) |> group_by(season) |>
  summarize(games = n(), upcoming = sum(is_upcoming),
            pct_features_filled = mean(as.matrix(across(all_of(setdiff(num_cols, "season")), ~ !is.na(.x)))))
print(cov)

write_rds(v36, here("objects", str_glue("modeling_data_{version}.rds")))
cat(sprintf("Saved objects/modeling_data_%s.rds: %d x %d\n", version, nrow(v36), ncol(v36)))
