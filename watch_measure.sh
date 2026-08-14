#!/bin/bash
# Fires when the composition-drift measurement saves or crashes.
cd "$(dirname "$0")"
while true; do
  if grep -q "\[saved\]" logs_pgram/measure.log 2>/dev/null; then
    echo "MEASUREMENT DONE"
    grep -E "^\[pgram_gramflow\] T19|^\[adam\] T19" logs_pgram/measure.log
    exit 0
  fi
  if grep -q "Traceback" logs_pgram/measure.log 2>/dev/null; then
    echo "MEASUREMENT CRASH"
    tail -5 logs_pgram/measure.log
    exit 0
  fi
  sleep 300
done
