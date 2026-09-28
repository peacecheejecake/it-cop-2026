#!/usr/bin/env bash
# For a train-codebert run started WITHOUT train_with_progress.py: riskbench only logs at
# epoch end, so in-epoch progress is estimated from the previous epoch's duration.
# Usage: watch_training.sh <train-log> [interval_s]
LOG="${1:?train log path}"; INTERVAL="${2:-30}"
start=$(stat -f %B "$LOG")
while :; do
  now=$(date +%s); last=$(stat -f %m "$LOG"); n=$(grep -c "epoch=" "$LOG")
  if [ "$n" -gt 0 ]; then per=$(( (last - start) / n )); else per=0; fi
  in_epoch=$(( now - (n > 0 ? last : start) ))
  pct="?"; eta="?"
  if [ "$per" -gt 0 ]; then pct=$(( 100 * in_epoch / per )); eta="$(( (per - in_epoch) / 60 ))m"; fi
  gpu=$(ioreg -r -d 1 -c IOAccelerator | grep -oE '"Device Utilization %"=[0-9]+' | cut -d= -f2)
  gmem=$(ioreg -r -d 1 -c IOAccelerator | grep -oE '"In use system memory"=[0-9]+' | cut -d= -f2)
  clear
  date '+%H:%M:%S'
  echo "epochs done: $n   epoch avg: $((per / 60))m   current epoch: $((in_epoch / 60))m (~${pct}%, ETA ${eta})"
  echo "GPU util: ${gpu}%   GPU mem: $((gmem / 1073741824)) GiB   $(sysctl -n vm.swapusage | awk '{print "swap used:", $6}')"
  echo; tail -5 "$LOG"
  grep -q "^exit=" "$LOG" && break
  sleep "$INTERVAL"
done
