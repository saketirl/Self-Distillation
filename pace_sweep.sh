#!/bin/bash
# Keep exactly MAXRUN sweep processes running; SIGSTOP the rest and SIGCONT
# them as slots free (most-advanced first). Exits when all have finished.
cd "$(dirname "$0")"
MAXRUN=1
PIDS="1025822 1025823 1037375 1037376 1037377"
progress() {  # tasks completed, from the pid's log
  local pid=$1
  local log=$(ls -l /proc/$pid/fd 2>/dev/null | grep -oP "logs_pgram/\S+\.log" | head -1)
  [ -n "$log" ] && grep -c "T[0-9]*:" "$log" 2>/dev/null || echo 0
}
while true; do
  alive=""; for p in $PIDS; do [ -d /proc/$p ] && alive="$alive $p"; done
  [ -z "$alive" ] && { echo "PACER DONE: all sweep processes finished"; exit 0; }
  running=""; stopped=""
  for p in $alive; do
    st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null)
    if [ "$st" = "T" ]; then stopped="$stopped $p"; else running="$running $p"; fi
  done
  nrun=$(echo $running | wc -w)
  if [ "$nrun" -gt "$MAXRUN" ]; then
    # stop the least-advanced extras
    for p in $(for q in $running; do echo "$(progress $q) $q"; done | sort -n | head -$((nrun - MAXRUN)) | awk '{print $2}'); do
      kill -STOP $p && echo "paused $p"
    done
  elif [ "$nrun" -lt "$MAXRUN" ] && [ -n "$stopped" ]; then
    for p in $(for q in $stopped; do echo "$(progress $q) $q"; done | sort -rn | head -$((MAXRUN - nrun)) | awk '{print $2}'); do
      kill -CONT $p && echo "resumed $p"
    done
  fi
  sleep 60
done
