#!/bin/bash
# Step-6T per-node runner: bash step6t_run_node.sh <seed> [<seed> ...]
# (AMENDMENT 2026-06-11; frozen artifacts 88be6ae, registered 2565466.)
# Builds both windowed corpora deterministically (builder seed 0), runs
# the G2 loader test, then for each seed trains both cells and scores
# offline (rvp31 per-template + mech_margins) — the step6_run_node.sh
# pipeline unchanged. Markers: STEP6T_BUILT, CELLDONE_<cell>_s<seed>,
# NODEDONE at the end.
set -euo pipefail
cd /workdir
export PYTHONPATH=/workdir/src:/workdir/scripts
PY=~/miniconda3/bin/python

# G2 pre-launch gate: windowed-loader unit test must pass on this node
$PY -m pytest src/fogen/training/test_windowed.py -q

build_cell() {
  local cell=$1 dose=$2
  if [ ! -f data/climbmix/bpe8192/${cell}/synth/rescue_manifest.json ]; then
    $PY scripts/gen_rescue_docs.py --dose "$dose" \
      --tokenizer-dir data/climbmix/bpe8192 \
      --corpus-manifest data/climbmix/bpe8192/shards/manifest.json \
      --out-dir data/climbmix/bpe8192/${cell}/synth
  fi
  mkdir -p data/climbmix/bpe8192/${cell}/shards
  ln -sf /workdir/data/climbmix/bpe8192/shards/shard_*.bin \
    data/climbmix/bpe8192/${cell}/shards/
  i=0
  for f in data/climbmix/bpe8192/${cell}/synth/shard_*.bin; do
    cp -f "$f" data/climbmix/bpe8192/${cell}/shards/shard_9$(printf %04d $i).bin
    i=$((i+1))
  done
}

# registered dose parameters: total dose 1.0 x delta_TS delivered within
# the exposure window -> synth dose = steps_total / window_steps
build_cell rescue_d100_early "$($PY -c 'print(4400/2400)')"
build_cell rescue_d100_late  2.75
echo STEP6T_BUILT

for s in "$@"; do
  for cell in rescue_d100_early rescue_d100_late; do
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
