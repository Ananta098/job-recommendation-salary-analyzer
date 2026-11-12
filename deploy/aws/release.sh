#!/bin/bash
# shellcheck disable=SC2154  # BUCKET, STACK and REGION are set by /opt/job-recommender/instance.env
# Install one release on the EC2 instance without risking the running app.
#   1. reuse the image if present, else load the copy saved in S3, else build it (dependency layers are cached)
#   2. smoke-test the image: load the bundle and rank a sample profile
#   3. start it next to the live app on 127.0.0.1:8502 and wait for its health URL
#   4. switch the service to it; if the switched app is unhealthy, restore the previous version
# Usage (root, on the instance): release.sh VERSION COMMIT
set -euo pipefail
VERSION=$1
COMMIT=${2:-unknown}
BASE=/opt/job-recommender
SRC=$BASE/releases/$VERSION
IMAGE=job-recommender:$VERSION
UNIT=/etc/systemd/system/job-recommender.service
CANDIDATE=job-recommender-candidate
# shellcheck source=/dev/null
. "$BASE/instance.env"
CACHE=s3://$BUCKET/$STACK/images/$VERSION.tar.gz

log() { echo "[release $(date -u +%H:%M:%S)] $*"; }
healthy() {  # healthy PORT ATTEMPTS  (5 seconds apart)
  local attempt
  for attempt in $(seq 1 "$2"); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$1/_stcore/health" >/dev/null 2>&1; then return 0; fi
    [ "$attempt" -lt "$2" ] && sleep 5
  done
  return 1
}
backup() { if [ -f "$1" ]; then cp -f "$1" "$1.previous"; else rm -f "$1.previous"; fi; }
restore() { if [ -f "$1.previous" ]; then mv -f "$1.previous" "$1"; else rm -f "$1"; fi; }
prune() {  # keep the three newest images plus the current and previous ones
  local tag keep=" $VERSION ${PREVIOUS:-} "
  for tag in $(docker images job-recommender --format '{{.Tag}}' | sort -r | tail -n +4); do
    case "$keep" in *" $tag "*) continue ;; esac
    docker rmi "job-recommender:$tag" >/dev/null 2>&1 || true
    rm -rf "${BASE:?}/releases/$tag"
  done
  docker image prune -f >/dev/null 2>&1 || true
}

exec 9>/run/job-recommender-release.lock
flock -n 9 || { log "another release is already running"; exit 1; }
trap 'docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true' EXIT

# 1. image
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "image $IMAGE is already on this server"
elif aws s3 ls "$CACHE" --region "$REGION" >/dev/null 2>&1 \
     && aws s3 cp "$CACHE" - --region "$REGION" --only-show-errors | gunzip | docker load >/dev/null; then
  log "loaded the saved copy of $IMAGE from S3"
else
  log "building $IMAGE"
  docker build --quiet -f "$SRC/Dockerfile.deploy" -t "$IMAGE" "$SRC" >/dev/null
  nohup setsid bash -c "docker save '$IMAGE' | gzip -1 | aws s3 cp - '$CACHE' --region '$REGION' --only-show-errors" \
    >>/var/log/job-recommender-image-cache.log 2>&1 &
  log "saving a copy of the image to S3 in the background (faster rollback and recovery)"
fi

# 2. smoke test
log "smoke test: loading the bundle and ranking a sample profile"
docker run --rm --read-only --tmpfs /tmp:rw,size=512m --cap-drop ALL --security-opt no-new-privileges \
  --entrypoint python "$IMAGE" -m recommender.smoke

# 3. candidate next to the live app
docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
docker run -d --name "$CANDIDATE" --read-only --tmpfs /tmp:rw,size=512m --cap-drop ALL \
  --security-opt no-new-privileges -p 127.0.0.1:8502:8501 "$IMAGE" >/dev/null
if ! healthy 8502 36; then
  log "the new version did not become healthy; the running version was not touched"
  docker logs --tail 40 "$CANDIDATE" 2>&1 || true
  exit 1
fi
docker rm -f "$CANDIDATE" >/dev/null
log "new version healthy next to the live app; switching"

# 4. switch (with automatic restore)
PREVIOUS=$(cat "$BASE/current_version" 2>/dev/null || true)
backup "$UNIT"
backup "$BASE/release.env"
install -m 0644 "$SRC/deploy/aws/job-recommender.service" "$UNIT"
printf 'VERSION=%s\nCOMMIT=%s\nDEPLOYED_AT=%s\n' "$VERSION" "$COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/release.env"
systemctl daemon-reload
systemctl enable job-recommender >/dev/null 2>&1
systemctl restart job-recommender
if healthy 8501 36; then
  echo "$VERSION" > "$BASE/current_version"
  if [ -n "$PREVIOUS" ] && [ "$PREVIOUS" != "$VERSION" ]; then echo "$PREVIOUS" > "$BASE/previous_version"; fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $VERSION $COMMIT" >> "$BASE/history"
  rm -f "$UNIT.previous" "$BASE/release.env.previous"
  prune
  log "now serving $VERSION"
  exit 0
fi
log "version $VERSION failed after the switch; restoring ${PREVIOUS:-the previous service}"
restore "$UNIT"
restore "$BASE/release.env"
systemctl daemon-reload
systemctl restart job-recommender || true
if healthy 8501 36; then log "previous version restored and healthy"; else log "WARNING: the previous version is not healthy either; run make aws-logs"; fi
exit 1
