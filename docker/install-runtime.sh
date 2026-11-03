#!/bin/sh
set -eu
apt-get update
apt-get install -y --no-install-recommends libgomp1
rm -rf /var/lib/apt/lists/*
useradd --create-home --uid 10001 app
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
