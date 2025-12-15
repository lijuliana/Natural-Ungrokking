#!/bin/bash
# Step-6 per-node runner: bash step6_run_node.sh <cell> [<cell> ...]
# For each cell: ensure its corpus exists (deterministic builders, seed
# 0), then for seeds 42 43 44: train -> offline rvp31 rescore
# (per-template) -> mech_margins. Markers: CELLDONE_<cell>_s<seed>,
# NODEDONE at the end.
set -euo pipefail
cd /workdir
export PYTHONPATH=/workdir/src:/workdir/scripts
PY=~/miniconda3/bin/python

build_cell() {
  case "$1" in
    rescue_d*)
      tag=${1#rescue_d}
      case $tag in 001) dose=0.01;; 010) dose=0.1;; 100) dose=1.0;; 300) dose=3.0;; esac
      if [ ! -f data/climbmix/bpe8192/rescue_d${tag}/synth/rescue_manifest.json ]; then
        $PY scripts/gen_rescue_docs.py --dose $dose \
          --tokenizer-dir data/climbmix/bpe8192 \
          --corpus-manifest data/climbmix/bpe8192/shards/manifest.json \
          --out-dir data/climbmix/bpe8192/rescue_d${tag}/synth
      fi
      mkdir -p data/climbmix/bpe8192/rescue_d${tag}/shards
      ln -sf /workdir/data/climbmix/bpe8192/shards/shard_*.bin \
        data/climbmix/bpe8192/rescue_d${tag}/shards/
      i=0
      for f in data/climbmix/bpe8192/rescue_d${tag}/synth/shard_*.bin; do
        cp -f "$f" data/climbmix/bpe8192/rescue_d${tag}/shards/shard_9$(printf %04d $i).bin
        i=$((i+1))
      done
      ;;
    kill_p*)
      tag=${1#kill_p}
      case $tag in 437) rate=0.437;; 645) rate=0.645;; 1000) rate=1.0;; esac
      if [ ! -f data/tinystories/bpe8192/kill_p${tag}/shards/manifest.json ]; then
        $PY scripts/build_kill_shards.py data/tinystories/bpe8192/shards \
          --tokenizer-dir data/tinystories/bpe8192 \
          --scan-json runs/gendered_tokens_ts.json \
          --flip-rate $rate \
          --out-dir data/tinystories/bpe8192/kill_p${tag}/shards
      fi
      ;;
  esac
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
