#!/bin/sh
# Read-only pre-deployment check. Builds nothing and contacts no cloud service.
#   sh deploy/preflight.sh            (run from anywhere)
#   PYTHON=.venv/bin/python sh deploy/preflight.sh
# Exit code 0 = ready for the Docker rehearsal and the AWS package step; 1 = something blocks deployment.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
fail=0
ok()   { printf 'OK    %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; fail=1; }
warn() { printf 'WARN  %s\n' "$1"; }

# 1. Python. The pinned packages (numpy 1.26, pandas 2.2) have no wheels for Python 3.13/3.14.
if "$PY" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
  ver=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  ok "Python $ver"
  [ "$ver" = "3.12" ] || warn "Python 3.12 is the tested version"
else
  found=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "none")
  bad "Python 3.11 or 3.12 required (found: $found). Use  make venv  or set PYTHON=/path/to/python3.12"
fi
if "$PY" -c 'import numpy, pandas, sklearn, torch, sentence_transformers, streamlit' 2>/dev/null; then
  ok "Python packages installed"
else
  warn "Python packages missing in this interpreter: run  make venv  (only needed to build or check locally; Docker has its own)"
fi

# 2. Model bundle and English translation assets (full SHA-256 verification)
if out=$("$PY" -m recommender.bundle check --deep 2>&1); then ok "$out"; else bad "$(printf '%s' "$out" | tail -n 1)"; fi
if out=$("$PY" -m recommender.translation check --deep 2>&1); then ok "$out"; else bad "$(printf '%s' "$out" | tail -n 1)"; fi

# Hugging Face key (optional: lifts download rate limits for the model build)
if [ -n "${HF_TOKEN:-}" ] || [ -f .hf_token ]; then ok "Hugging Face key found"; else warn "No Hugging Face key (.hf_token): optional, run  make hf-key"; fi

# 3. Files the deployment image copies
for f in Dockerfile.deploy requirements-app.txt requirements-app.lock.txt app.py .streamlit/config.toml \
         docker/entrypoint.sh docker/install-runtime.sh; do
  if [ -f "$f" ]; then ok "$f"; else bad "missing $f"; fi
done
if [ -f .env.deploy ]; then ok ".env.deploy"; else warn ".env.deploy not found: run  cp .env.deploy.example .env.deploy"; fi

# 4. Docker (needed only for the local rehearsal; EC2 builds the image itself)
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker daemon reachable"
    envopt=""; [ -f .env.deploy ] && envopt="--env-file .env.deploy"
    # shellcheck disable=SC2086
    if docker compose $envopt -f compose.deploy.yaml config -q 2>/dev/null; then ok "compose.deploy.yaml parses"; else bad "compose.deploy.yaml does not parse"; fi
  else
    warn "Docker is installed but the daemon is not running: start Docker Desktop before the rehearsal"
  fi
else
  warn "Docker not installed: the local rehearsal cannot run (the EC2 route does not need it)"
fi

# 5. AWS settings (only needed for the cloud step)
if [ -f .env.aws ]; then
  if grep -q 'PASTE_' .env.aws; then warn ".env.aws still has PASTE_ placeholders: put your AWS keys in it before  make aws-deploy"
  else ok ".env.aws filled in"; fi
else
  warn ".env.aws not found: run  make aws-deploy  once to create it, then paste your AWS keys"
fi

echo
if [ "$fail" -eq 0 ]; then echo "Preflight passed. Next: the Docker rehearsal in docs/PROJECT_HANDBOOK.md, section 4."; else echo "Preflight FAILED. Fix the FAIL lines above first."; fi
exit "$fail"
