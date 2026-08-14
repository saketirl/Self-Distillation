#!/bin/bash
# Monitor loop: emits one line when the sweep completes or errors, else silent.
cd "$(dirname "$0")"
while true; do
  out=$(bash check_sweep.sh)
  if echo "$out" | grep -qE "SWEEP COMPLETE|SWEEP ERRORS"; then
    echo "$out" | head -14
    exit 0
  fi
  sleep 600
done
