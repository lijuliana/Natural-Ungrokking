#!/usr/bin/env bash
# Run the borrowed mechanism analyses (DECISIONS.md 2026-06-11) over a
# list of run dirs on a worker node, sync outputs to S3, drop a marker.
#   bash scripts/mech_borrowed_node.sh runs/v1_repro_s42 runs/...
# Requires cwd=/workdir (relative tokenizer/battery paths in configs).
set -uo pipefail
cd /workdir
PY=/home/ubuntu/miniconda3/bin/python
LOG=/workdir/runs/mech_borrowed_node.log
fail=0
{
  echo "=== mech_borrowed start $(date -u +%FT%TZ) runs: $*"
  for run in "$@"; do
    for job in mech_heads mech_patch bootstrap_cis; do
      echo "--- $job $run $(date -u +%FT%TZ)"
      if ! nice -n 10 $PY scripts/$job.py "$run"; then
        echo "FAILED $job $run"; fail=1; continue
      fi
    done
    aws s3 cp "$run/mech_heads.jsonl" "s3://fogen-phase/$run/" || fail=1
    aws s3 cp "$run/mech_patch.json" "s3://fogen-phase/$run/" || fail=1
  done
  aws s3 cp runs/bootstrap_cis.json \
      "s3://fogen-phase/runs/bootstrap_cis_$(hostname).json" || fail=1
  if [ "$fail" -eq 0 ]; then
    touch /workdir/runs/MECHDONE_"$(hostname)"
    echo "=== MECHDONE $(date -u +%FT%TZ)"
  else
    echo "=== finished WITH FAILURES $(date -u +%FT%TZ)"
  fi
} >> "$LOG" 2>&1
