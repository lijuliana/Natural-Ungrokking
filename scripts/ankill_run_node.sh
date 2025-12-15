#!/bin/bash
# a/an-kill per-node runner: bash ankill_run_node.sh <cell> [<cell> ...]
# Cells: ankill_p500 ankill_p667 ankill_p750 ankill_p900 ankill_p1000.
# For each cell: build its corpus if absent (deterministic, builder rng
# seed 0; cross-node manifest sha256s must match), then for seeds
# 42 43 44: train -> offline rvp31 rescore (per-template) ->
# mech_margins. Markers: CELLDONE_<cell>_s<seed>, NODEDONE at the end.
set -euo pipefail
cd /workdir
export PYTHONPATH=/workdir/src:/workdir/scripts
PY=~/miniconda3/bin/python

build_cell() {
  tag=${1#ankill_p}
  case $tag in 500) rate=0.5;; 667) rate=0.667;; 750) rate=0.75;;
               900) rate=0.9;; 1000) rate=1.0;;
               *) echo "unknown cell $1" >&2; exit 1;; esac
  if [ ! -f data/tinystories/bpe8192/ankill_p${tag}/shards/manifest.json ]; then
    $PY scripts/build_an_kill_shards.py data/tinystories/bpe8192/shards \
      --tokenizer-dir data/tinystories/bpe8192 \
      --flip-rate $rate \
      --out-dir data/tinystories/bpe8192/ankill_p${tag}/shards
  fi
}

for cell in "$@"; do
  build_cell "$cell"
  echo "BUILT_${cell}"
  for s in 42 43 44; do
    if [ ! -f runs/${cell}_s${s}/probe_log_rvp31.jsonl ]; then
      $PY -m fogen.training.train --config configs/${cell}.yaml --seed $s
      $PY scripts/score_ckpts.py runs/${cell}_s${s} \
        --battery data/probes/rvp3/battery.jsonl --tag rvp31 --per-template
      $PY scripts/mech_margins.py runs/${cell}_s${s}
    fi
    echo "CELLDONE_${cell}_s${s}"
  done
done
echo NODEDONE
