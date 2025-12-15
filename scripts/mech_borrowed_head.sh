#!/usr/bin/env bash
# Head-node variant of mech_borrowed_node.sh: run the borrowed
# mechanism analyses (DECISIONS.md 2026-06-11) over a group of runs,
# streaming checkpoints from S3 (head node is idle while the a/an
# ladder trains; registered compute plan allows this).
#   bash scripts/mech_borrowed_head.sh <group> runs/v1_repro_s42 ...
# cwd must be the repo root (~/fogen-icml).
set -uo pipefail
GROUP=$1; shift
PY="$HOME/venv/bin/python"
LOG="runs/mech_borrowed_${GROUP}.log"
fail=0
{
  echo "=== mech_borrowed[$GROUP] start $(date -u +%FT%TZ) runs: $*"
  for run in "$@"; do
    for job in mech_heads mech_patch; do
      echo "--- $job $run $(date -u +%FT%TZ)"
      if ! OMP_NUM_THREADS=1 $PY scripts/$job.py "$run"; then
        echo "FAILED $job $run"; fail=1; continue
      fi
    done
    echo "--- bootstrap_cis $run $(date -u +%FT%TZ)"
    OMP_NUM_THREADS=1 $PY scripts/bootstrap_cis.py "$run" \
        --out "runs/bootstrap_cis_${GROUP}.json" \
        || { echo "FAILED bootstrap_cis $run"; fail=1; }
    aws s3 cp "$run/mech_heads.jsonl" "s3://fogen-phase/$run/" || fail=1
    aws s3 cp "$run/mech_patch.json" "s3://fogen-phase/$run/" || fail=1
  done
  aws s3 cp "runs/bootstrap_cis_${GROUP}.json" \
      "s3://fogen-phase/runs/" || fail=1
  if [ "$fail" -eq 0 ]; then
    touch "runs/MECHDONE_${GROUP}"
    echo "=== MECHDONE[$GROUP] $(date -u +%FT%TZ)"
  else
    echo "=== [$GROUP] finished WITH FAILURES $(date -u +%FT%TZ)"
  fi
} >> "$LOG" 2>&1
