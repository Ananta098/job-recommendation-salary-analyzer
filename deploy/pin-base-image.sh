#!/bin/sh
# Pin Dockerfile.deploy to the current digest of python:3.12-slim-bookworm (make pin-base).
# Needs only curl and python3 (no Docker). Re-run deliberately to take base-image security updates.
set -eu
cd "$(dirname "$0")/.."
REPO=library/python
TAG=3.12-slim-bookworm
token=$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
digest=$(curl -fsSI -H "Authorization: Bearer $token" \
  -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json" \
  "https://registry-1.docker.io/v2/$REPO/manifests/$TAG" | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:" {print $2}')
case "$digest" in
  sha256:*) ;;
  *) echo "Could not resolve the digest of python:$TAG (no internet access to Docker Hub?)"; exit 1 ;;
esac
sed -i.bak -E "s#^FROM python:3\.12-slim-bookworm(@sha256:[0-9a-f]{64})?#FROM python:3.12-slim-bookworm@$digest#" Dockerfile.deploy
rm -f Dockerfile.deploy.bak
echo "Pinned Dockerfile.deploy to python:$TAG@$digest"
