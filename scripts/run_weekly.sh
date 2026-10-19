#!/bin/bash
# Weekly update: pull the latest data, rebuild features, log + grade paper picks.
# Best run Saturday morning (final injury reports are out Friday afternoon);
# running again later just updates picks for games that haven't kicked off.
#
#   bash scripts/run_weekly.sh
set -euo pipefail
cd "$(dirname "$0")/.."
log=results/run_weekly.log
: > "$log"

step() { echo "== $1"; shift; "$@" >> "$log" 2>&1 || { echo "FAILED — see $log"; exit 1; }; }

step "Pull 2026 data, lines, weather"   Rscript scripts/02_prepare_recent_data.R
step "Base features (01)"              Rscript scripts/01_build_model_data.R
step "Advanced features (10)"          Rscript scripts/10_build_advanced_features.R
step "Python feature store"            /usr/bin/python3 scripts/run_python_features.py
step "Merge (19c)"                     Rscript scripts/19c_rebuild_v3_clean.R
step "Final table (20)"                env MODEL_VERSION=v37 Rscript scripts/20_finalize_modeling_table.R
echo "== Injury tracker"
Rscript scripts/53_injury_tracker.R 2>>"$log" || { echo "FAILED — see $log"; exit 1; }
echo "== Scenario tracker"
Rscript scripts/55_scenario_tracker.R 2>>"$log" || { echo "FAILED — see $log"; exit 1; }
