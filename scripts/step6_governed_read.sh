#!/bin/bash
# Step-6 GOVERNED READ runbook.
#
# DO NOT RUN before all three node drivers report NODEDONE. The driver
# (scripts/step6_run_node.sh) only ECHOES "NODEDONE" as the final line
# of /workdir/logs/step6_node.log -- it never creates a marker file --
# so the gate checks NODEDONE presence plus direct artifact existence.
# This script performs the one-shot governed read: it pulls the frozen
# per-run artifacts, runs the frozen evaluator AND the independent
# verifier, asserts they agree on all 19 verdict keys, and builds the
# overlay figure. Any evaluator/verifier disagreement is adjudicated
# against the registered text and logged -- never patched silently.
#
# KILLS ARE PULLED FROM GRID2 ONLY: grid3 holds a stale kill_p437
# artifact from an aborted pre-swap launch and must not be a source.
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="$HOME/.ssh/fogen-key.pem"
HEAD="ubuntu@44.201.28.18"
FILES="probe_log_rvp31.jsonl probe_log_rvp31_templates.jsonl mech_margins.jsonl"

# note: stale duplicate drivers on grid2 AND grid3 (incidents logged
# 2026-06-11) corrupted the shared node logs -- grid2's flushed buffered
# crash output after the real NODEDONE, and grid3's truncated the log so
# only the final cell's CELLDONE survives. Log markers are therefore
# unreliable; the gate verifies completion directly: (a) NODEDONE
# present anywhere in the log, (b) every expected per-run artifact
# present and non-empty, (c) no live train/score process on the node.
S=""; for s in 42 43 44; do S="$S $s"; done
runs_for() { local c; for c in "$@"; do for s in $S; do echo "${c}_s${s}"; done; done; }
cells_for() { case $1 in
  dev)   echo "rescue_d100 rescue_d300";;
  grid2) echo "kill_p437 kill_p645 kill_p1000";;
  grid3) echo "rescue_d001 rescue_d010";;
esac; }

for node in dev grid2 grid3; do
  # shellcheck disable=SC2046
  if ! ssh -i "$KEY" "$HEAD" "ssh $node bash -s" <<EOF
fail=0
grep -aq NODEDONE /workdir/logs/step6_node.log || { echo "NO NODEDONE on $node"; fail=1; }
for r in $(runs_for $(cells_for $node) | xargs); do
  for f in $FILES; do
    [ -s "/workdir/runs/\$r/\$f" ] || { echo "MISSING $node:\$r/\$f"; fail=1; }
  done
done
exit \$fail
EOF
  then
    echo "ABORT: $node not done; governed read not allowed yet"
    exit 1
  fi
  # ps output grepped locally so the remote cmdline can't self-match
  live=$(ssh -i "$KEY" -n "$HEAD" "ssh -n $node ps aux" \
         | grep -cE '[f]ogen\.training|[s]core_ckpts|[m]ech_margins' || true)
  if [ "$live" -ne 0 ]; then
    echo "ABORT: $node has $live live train/score processes"
    exit 1
  fi
done

pull() {
  local node=$1; shift
  for r in "$@"; do
    mkdir -p "runs/$r"
    for f in $FILES; do
      if [ -s "runs/$r/$f" ]; then continue; fi
      echo "pull $node:$r/$f"
      ssh -i "$KEY" -n "$HEAD" "ssh -n $node cat /workdir/runs/$r/$f" \
        > "runs/$r/$f" || { echo "MISSING $node:$r/$f"; exit 1; }
    done
  done
}

# shellcheck disable=SC2046
pull grid2 $(runs_for kill_p437 kill_p645 kill_p1000)          # KILLS: grid2 ONLY
# shellcheck disable=SC2046
pull grid3 $(runs_for rescue_d001 rescue_d010)
# shellcheck disable=SC2046
pull dev   $(runs_for rescue_d100 rescue_d300)

# Base cells come from the canonical S3 mirrors (Step-3/5 artifacts,
# already read under earlier gates); dev's local copies are partial.
for r in $(runs_for v1_repro web_packed_v2); do
  mkdir -p "runs/$r"
  for f in $FILES; do
    if [ -s "runs/$r/$f" ]; then continue; fi
    echo "pull s3:$r/$f"
    ssh -i "$KEY" -n "$HEAD" "aws s3 cp s3://fogen-phase/runs/$r/$f -" \
      > "runs/$r/$f" || { echo "MISSING s3:$r/$f"; exit 1; }
    [ -s "runs/$r/$f" ] || { echo "EMPTY s3:$r/$f"; exit 1; }
  done
done

python3 scripts/eval_step6.py   --runs-root runs --seeds 42 43 44 --out runs/eval_step6.json
python3 scripts/verify_step6.py --runs-root runs --seeds 42 43 44 --out runs/verify_step6.json

python3 - <<'EOF'
import json, sys
a = json.load(open("runs/eval_step6.json"))
b = json.load(open("runs/verify_step6.json"))
keys = sorted(set(a["verdicts"]) | set(b["verdicts"]))
bad = [k for k in keys if a["verdicts"].get(k) != b["verdicts"].get(k)]
print(f"verdict keys compared: {len(keys)}")
if bad:
    for k in bad:
        print(f"DISAGREE {k}: eval={a['verdicts'].get(k)} "
              f"verify={b['verdicts'].get(k)}")
    sys.exit("EVALUATOR/VERIFIER DISAGREEMENT -- adjudicate against "
             "registered text, log in RESEARCH_LOG, do not proceed")
print("evaluator and verifier AGREE on all verdict keys")
EOF

python3 scripts/fig_step6_overlay.py --eval runs/eval_step6.json
echo "GOVERNED READ COMPLETE -- log the read in RESEARCH_LOG before anything else"
