#!/bin/bash
# Step-6 corpus build + pre-launch gates (run on a node from /workdir).
# Deterministic (all builders rng seed 0) so nodes can rebuild identical
# artifacts independently; cross-node manifest sha256s must match.
set -euo pipefail
cd /workdir
export PYTHONPATH=/workdir/src:/workdir/scripts
PY=~/miniconda3/bin/python

for d in 001:0.01 010:0.1 100:1.0 300:3.0; do
  tag=${d%%:*}; dose=${d##*:}
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
  echo "RESCUE_d${tag}_READY"
done

for p in 437:0.437 645:0.645 1000:1.0; do
  tag=${p%%:*}; rate=${p##*:}
  if [ ! -f data/tinystories/bpe8192/kill_p${tag}/shards/manifest.json ]; then
    $PY scripts/build_kill_shards.py data/tinystories/bpe8192/shards \
      --tokenizer-dir data/tinystories/bpe8192 \
      --scan-json runs/gendered_tokens_ts.json \
      --flip-rate $rate \
      --out-dir data/tinystories/bpe8192/kill_p${tag}/shards
  fi
  echo "KILL_p${tag}_READY"
done

# instrument verification: post-kill GIRLS_V2 counter ratios
for tag in 437 645 1000; do
  $PY scripts/count_support.py data/tinystories/bpe8192/kill_p${tag}/shards \
    --tokenizer-dir data/tinystories/bpe8192 --mode window \
    --out runs/support_kill_p${tag}_window.json
done
echo "KILL_COUNTER_DONE"

# hard-fail disjointness gate (D4 on every kill manifest)
for tag in 437 645 1000; do
  $PY scripts/check_intervention_disjoint.py \
    --rescue-dirs data/climbmix/bpe8192/rescue_d001/synth \
                  data/climbmix/bpe8192/rescue_d010/synth \
                  data/climbmix/bpe8192/rescue_d100/synth \
                  data/climbmix/bpe8192/rescue_d300/synth \
    --kill-manifest data/tinystories/bpe8192/kill_p${tag}/shards/manifest.json \
    --out runs/step6_disjointness_p${tag}.json
done
echo "STEP6_BUILD_OK"
