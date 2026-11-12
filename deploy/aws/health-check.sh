#!/bin/bash
# Runs every minute (job-recommender-health.timer): restart the app after 3 failed health checks in a row.
# Never fails the timer.
COUNT=/run/job-recommender-health-failures
# A release is switching versions: do not interfere.
if ! flock -n /run/job-recommender-release.lock true 2>/dev/null; then exit 0; fi
if curl -fsS --max-time 10 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  echo 0 > "$COUNT"
else
  failures=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
  echo "$failures" > "$COUNT"
  if [ "$failures" -ge 3 ]; then
    logger -t job-recommender "health check failed $failures times; restarting the app"
    systemctl restart job-recommender || true
    echo 0 > "$COUNT"
  fi
fi
exit 0
