#!/bin/sh
set -eu
apt-get update
# libgomp1 comes from the Debian bookworm archive of the pinned base image. Its exact version is not
# pinned because Debian removes superseded versions, which would break future builds; Python packages are pinned.
apt-get install -y --no-install-recommends libgomp1
rm -rf /var/lib/apt/lists/*
useradd --create-home --uid 10001 app
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/app.txt
