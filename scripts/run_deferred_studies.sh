#!/usr/bin/env bash
# Kicks off every deferred study from brief-2 back-to-back.
#
# Each block writes to its own CSV incrementally (resume-safe: killing the
# script and restarting it picks up where any block left off, thanks to the
# scripts' _load_existing dedup). Total expected wall-clock: 5-8 h on the
# reference server (2x V100, 56 cores, 121 GB RAM) at 16 parallel workers
# for the studies that support it.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

WORKERS=${WORKERS:-16}
FIGDIR="docs/figures"

log() { printf "\n===== %s =====\n" "$*" ; }

log "B1a Fig 6.3 budget-split (~30 min @ ${WORKERS} workers)"
python experiments/ch06_ppo_oracle/run_budget_split.py \
    --workers "$WORKERS" --figdir "$FIGDIR"

log "B1b Fig 6.1 BR-quality (~15 min @ ${WORKERS} workers)"
python experiments/ch06_ppo_oracle/run_br_quality.py \
    --workers "$WORKERS" --figdir "$FIGDIR"

log "B4 Fig 12.1 warm-start ablation (~50 min @ ${WORKERS} workers)"
python experiments/ch12_shootout/run_warm_start.py \
    --workers "$WORKERS" --figdir "$FIGDIR"

log "B2/C2 Fig 7.3 Leduc dashboard + N=3 seed bands (~3 h serial per seed)"
python experiments/ch07_kuhn_leduc/run_ch07.py \
    --n-seeds 3 --with-dashboard --figdir "$FIGDIR"

log "C1a Ch 6 grid at N=5 (~2 h serial)"
python experiments/ch06_ppo_oracle/run_ch06.py \
    --n-seeds 5 --figdir "$FIGDIR"

log "C1b Ch 8 zoo at N=5 (~50 min serial)"
python experiments/ch08_meta_solvers/run_ch08.py \
    --n-seeds 5 --figdir "$FIGDIR"

log "C1c Ch 11 ablation at N=5 (~1.5 h serial)"
python experiments/ch11_diversity/run_ch11.py \
    --n-seeds 5 --figdir "$FIGDIR"

log "ALL STUDIES COMPLETE"
