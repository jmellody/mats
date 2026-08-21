#!/usr/bin/env bash
# Archive superseded scripts. Live files stay flat at the top level.
#
#   bash reorganize.sh          # dry run
#   bash reorganize.sh --go     # do it
#
# Commit first -- git history is what preserves the evolution of the project.
#
# WHY FLAT: every script imports config, acts, probe, logs from the top level.
# Moving them into subdirectories breaks that, because Python puts the SCRIPT's
# directory on sys.path, not the working directory. Sixteen live files at the
# root is not clutter; broken imports across six folders is.
#
# Nothing is deleted. archive/v1 is worth keeping: that pipeline is the
# negative case that motivates the construct-validity check, and a reviewer
# should be able to see it.

set -e
GO=false
[ "$1" = "--go" ] && GO=true

move () {
  dir="$1"; shift
  for f in "$@"; do
    [ -e "$f" ] || continue
    if $GO; then
      mkdir -p "$dir"
      git mv "$f" "$dir/" 2>/dev/null || mv "$f" "$dir/"
      echo "  moved       $f -> $dir/"
    else
      echo "  would move  $f -> $dir/"
    fi
  done
}

echo "=== archive/v1: the truth-detector pipeline ==="
echo "    (kept -- it is the negative case the whole project turns on)"
move archive/v1 make_stimuli.py rate_pairs.py dose.py contradiction.py order.py

echo
echo "=== archive/superseded: dead ends and earlier versions ==="
move archive/superseded run.py rq1.py neutral_baseline.py final_stats.py \
     resweep.py fast_extract.py add_wrappers.py diagnose.py extract.py \
     judge_outputs.py robustness.py

cat <<'EOF'

=== STAYS AT TOP LEVEL ===

  core          config.py  acts.py  probe.py  logs.py
  v2 pipeline   stimuli_v2.py  probe_v2.py  dynamics_v2.py  conflict.py
  validation    construct.py  ceiling.py  orthogonalize.py
  steering      steer.py  steer_check.py  kl_seq.py  kl_multilayer.py
                judge_local.py
  tooling       smoke.py  run_all.py  figures.py  vast_setup.sh
  docs          README.md  RESULTS.md  requirements.txt

Twenty files, each doing one thing.

=== ONE THING TO CHECK AFTER MOVING ===

run_all.py calls dose.py and contradiction.py, which are now archived. Either
point it at dynamics_v2.py or delete the stages -- it is a v1 orchestrator and
the v2 pipeline is run by hand.
EOF
