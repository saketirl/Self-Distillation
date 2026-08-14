#!/bin/bash
# One status probe for the P-Gram sweep: prints COMPLETE/ERRORS/progress lines.
cd "$(dirname "$0")"
done_n=$(grep -l "\[saved\]" logs_pgram/*.log 2>/dev/null | wc -l)
err=$(grep -l "Traceback" logs_pgram/*.log 2>/dev/null | wc -l)
div=$(grep -l "DIVERGED" logs_pgram/*.log 2>/dev/null | wc -l)
if [ "$err" -gt 0 ]; then
  echo "SWEEP ERRORS in: $(grep -l Traceback logs_pgram/*.log | tr '\n' ' ')"
  grep -h -A2 Traceback $(grep -l Traceback logs_pgram/*.log | head -1) | tail -3
elif [ "$done_n" -ge 9 ]; then
  echo "SWEEP COMPLETE (diverged: $div)"
  grep -h "done=" logs_pgram/*.log
else
  echo "progress: $done_n/9 done, $div diverged"
  for f in logs_pgram/*.log; do echo "$(basename $f .log): $(grep -c 'T[0-9]*:' $f) tasks"; done | head -9
fi
