#!/bin/bash
# shellcheck disable=SC2154  # BUCKET, STACK and REGION are set by /opt/job-recommender/instance.env
# Prepare the instance for a release (helper scripts, health timer, CloudWatch agent), then run release.sh.
# Safe to run again for every update or rollback. Usage (root, on the instance): install-host.sh VERSION COMMIT
set -euo pipefail
VERSION=$1
COMMIT=${2:-unknown}
BASE=/opt/job-recommender
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
. "$BASE/instance.env"
mkdir -p "$BASE/bin"
install -m 0755 "$HERE/release.sh" "$HERE/health-check.sh" "$BASE/bin/"
install -m 0644 "$HERE/job-recommender-health.service" "$HERE/job-recommender-health.timer" /etc/systemd/system/
systemctl daemon-reload

# CloudWatch agent: ships logs and memory/disk metrics. Optional: the app runs without it.
CTL=/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl
[ -x "$CTL" ] || dnf install -y -q amazon-cloudwatch-agent >/dev/null 2>&1 || true
if [ -x "$CTL" ]; then
  sed "s/__STACK__/$STACK/g" "$HERE/cloudwatch-agent.json" > "$BASE/cloudwatch-agent.json"
  if "$CTL" -a fetch-config -m ec2 -s -c "file:$BASE/cloudwatch-agent.json" >/dev/null 2>&1; then
    echo "CloudWatch agent running (logs and memory/disk metrics)"
  else
    echo "CloudWatch agent could not start; continuing without log shipping"
  fi
else
  echo "CloudWatch agent not available; continuing without log shipping"
fi

"$BASE/bin/release.sh" "$VERSION" "$COMMIT"
systemctl enable --now job-recommender-health.timer >/dev/null 2>&1 || true
