#!/bin/bash
# shellcheck disable=SC2154  # BUCKET, STACK and REGION are set by /opt/job-recommender/instance.env
# Runs every minute (job-recommender-health.timer): restart the app after 3 failed health checks in a row
# and publish JobRecommender/AppHealthy (1 or 0) for the CloudWatch alarm. Never fails the timer.
BASE=/opt/job-recommender
COUNT=/run/job-recommender-health-failures
# A release is switching versions: do not interfere.
if ! flock -n /run/job-recommender-release.lock true 2>/dev/null; then exit 0; fi
if curl -fsS --max-time 10 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  value=1
  echo 0 > "$COUNT"
else
  value=0
  failures=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
  echo "$failures" > "$COUNT"
  if [ "$failures" -ge 3 ]; then
    logger -t job-recommender "health check failed $failures times; restarting the app"
    systemctl restart job-recommender || true
    echo 0 > "$COUNT"
  fi
fi
if [ -f "$BASE/instance.env" ]; then
  # shellcheck source=/dev/null
  . "$BASE/instance.env"
  aws cloudwatch put-metric-data --region "$REGION" --namespace JobRecommender --metric-name AppHealthy \
    --dimensions "Stack=$STACK" --value "$value" >/dev/null 2>&1 || true
fi
exit 0
