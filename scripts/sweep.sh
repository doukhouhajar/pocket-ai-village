#!/usr/bin/env bash
set -euo pipefail
SEEDS="${SEEDS:-0 1 2 3 4}"
PAR="${PAR:-3}"   # runs in flight at once; vLLM batches their requests
phase="${1:-all}"
jobs=()
EXTRA="${EXTRA:-}"  #extra overrides for the task phases
add() { for s in $SEEDS; do jobs+=("$1 --seed $s --set $2 $EXTRA"); done; }
if [[ $phase == commons   || $phase == all ]]; then SAVE=$EXTRA; EXTRA=""; add commons "comm=true"; add commons "comm=false"; EXTRA=$SAVE; fi
if [[ $phase == receipts  || $phase == all ]]; then add receipts "audit=false"; add receipts "audit=true"; fi
if [[ $phase == principal || $phase == all ]]; then add principal "budget=10";  add principal "budget=3"; fi
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c 'python -m thevillage.run {} > /dev/null && echo "done: {}"'
python -m analysis.summarize
