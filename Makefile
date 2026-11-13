# Helpers for the local Docker run. Type `make help`.
SHELL := /bin/bash
PY312 ?= $(shell command -v python3.12 2>/dev/null || echo /opt/homebrew/opt/python@3.12/bin/python3.12)
VENV  ?= .venv
PY    := $(VENV)/bin/python
RAW   ?= datasets/HSE_dataset(D4).csv
ifndef RUN
RUN := artifacts/runs/run-$(shell date +%Y%m%d-%H%M%S)
endif
COMPOSE := docker compose --env-file .env.deploy -f compose.deploy.yaml

.PHONY: dev-tools test lint image-test ci pin-base aws-update aws-rollback aws-versions aws-status aws-logs aws-tools help hf-key venv bundle translation check env perms up down logs local-docker aws-plan aws-deploy aws-destroy

help:
	@echo "make hf-key       save your Hugging Face key to .hf_token (git-ignored, optional)"
	@echo "make venv         create .venv (Python 3.12) and install every requirement"
	@echo "make bundle       build artifacts/deployment from the raw dataset (skipped if it exists)"
	@echo "make translation  download the English-UI models into artifacts/translation"
	@echo "make check        read-only readiness check (deploy/preflight.sh)"
	@echo "make up           build the deployment image and start it on http://127.0.0.1:8501"
	@echo "make down|logs    stop it | show its log"
	@echo "make local-docker everything above in order: venv, bundle, translation, check, up"
	@echo "make test         unit and smoke tests (no dataset, no AWS)"
	@echo "make ci           test + shellcheck + cfn-lint + image build (same checks as GitHub Actions)"
	@echo "make pin-base     pin the Docker base image to its current digest"
	@echo "make aws-plan     dry run: bundle check, plan and cost estimate; changes nothing"
	@echo "make aws-deploy   first deployment to AWS EC2 using the keys in .env.aws"
	@echo "make aws-update   install a new version on the running server (switched only if healthy)"
	@echo "make aws-rollback back to the previous version (VERSION=<name> for a specific one)"
	@echo "make aws-versions | aws-status | aws-logs | aws-destroy"

hf-key:
	@read -r -s -p "Paste your Hugging Face token (hidden): " t; echo; \
	 [ -n "$$t" ] || { echo "Nothing entered."; exit 1; }; \
	 printf '%s\n' "$$t" > .hf_token; chmod 600 .hf_token; echo "Saved to .hf_token (git-ignored)."

venv:
	@test -x "$(PY312)" || { echo "Python 3.12 not found. Install it with:  brew install python@3.12"; exit 1; }
	@test -x $(PY) || "$(PY312)" -m venv $(VENV)
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' || { echo "$(VENV) is not Python 3.12. Run  rm -rf $(VENV)  and then  make venv  again."; exit 1; }
	$(PY) -m pip install --no-cache-dir --upgrade pip
	$(PY) -m pip install --no-cache-dir -r requirements.txt
	$(PY) -m ipykernel install --user --name jobrec --display-name "Python (jobrec)"
	$(PY) -m pip check

bundle:
	@if [ -f artifacts/deployment/manifest.json ]; then echo "artifacts/deployment already exists: skipping (delete it to rebuild)."; exit 0; fi; \
	 test -x $(PY) || { echo "Run  make venv  first."; exit 1; }; \
	 test -f "$(RAW)" || { echo "Raw dataset missing: $(RAW)"; echo "Put HSE_dataset(D4).csv into datasets/ (see datasets/README.md), then run make bundle again."; exit 1; }; \
	 if [ -f clean/d4_clean.csv ] && ls clean/sbert_test_*.npy >/dev/null 2>&1; then REUSE=--reuse-prepared; \
	   echo "Reusing the prepared data in clean/ (fast)."; else REUSE=; \
	   echo "No prepared data in clean/ (fresh clone): full rebuild from the raw dataset. This can take an hour or more."; fi; \
	 $(PY) -m recommender.pipeline run --run-dir "$(RUN)" --raw "$(RAW)" $$REUSE && \
	 mkdir -p artifacts && cp -R "$(RUN)/artifacts/deployment" artifacts/deployment && \
	 $(PY) -m recommender.bundle check --deep

translation:
	@test -f artifacts/deployment/manifest.json || { echo "Build the model bundle first:  make bundle"; exit 1; }
	$(PY) -m recommender.translation prepare
	$(PY) -m recommender.translation check --deep

check:
	PYTHON=$(PY) sh deploy/preflight.sh

env:
	@test -f .env.deploy || { cp .env.deploy.example .env.deploy; echo "Created .env.deploy"; }

# The container runs as user 10001, not as you: make sure it can read the project files and the bundle.
perms:
	@chmod -R a+rX app.py recommender m9_skills .streamlit docker artifacts 2>/dev/null || true

up: env perms
	@docker info >/dev/null 2>&1 || { echo "Docker Desktop is not running. Open it, wait until it says it is running, then run make up again."; exit 1; }
	$(COMPOSE) up --build -d
	@PORT=$$(grep -s '^DEPLOY_PORT=' .env.deploy | cut -d= -f2); PORT=$${PORT:-8501}; \
	 echo "Waiting for the app on port $$PORT (the first start loads the models) ..."; \
	 for i in $$(seq 1 60); do \
	   if curl -fsS "http://127.0.0.1:$$PORT/_stcore/health" >/dev/null 2>&1; then echo "Healthy. Open http://127.0.0.1:$$PORT and submit a profile to test the model."; exit 0; fi; sleep 5; done; \
	 echo "Not healthy after 5 minutes. Run  make logs"; exit 1

down: env
	$(COMPOSE) down

logs: env
	$(COMPOSE) logs --tail=100 app

local-docker: venv bundle translation check up

# ---- AWS (EC2). The only thing you fill in is .env.aws ----
.env.aws:
	@cp .env.aws.example .env.aws; chmod 600 .env.aws; \
	 echo "Created .env.aws. Open it, paste your AWS access key and secret key, save, then run this command again."; exit 1

aws-tools:
	@test -x $(PY) || { echo "Run  make venv  first."; exit 1; }
	@$(PY) -c 'import boto3' 2>/dev/null || $(PY) -m pip install --no-cache-dir -q -r requirements-deploy.txt

aws-plan: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py

aws-deploy: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --apply

aws-destroy: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --destroy

aws-update: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --update

aws-rollback: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --rollback $(VERSION)

aws-versions: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --versions

aws-status: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --status

MINUTES ?= 30
aws-logs: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --logs --minutes $(MINUTES)

# ---- Tests and CI: no raw dataset and no AWS needed ----
dev-tools:
	@test -x $(PY) || { echo "Run  make venv  first."; exit 1; }
	@$(PY) -c 'import pytest, cfnlint, shellcheck_py' 2>/dev/null || $(PY) -m pip install --no-cache-dir -q -r requirements-dev.txt

test: dev-tools
	$(PY) -m pytest -q tests

lint: dev-tools
	$(VENV)/bin/shellcheck docker/*.sh deploy/preflight.sh deploy/pin-base-image.sh deploy/aws/*.sh
	$(VENV)/bin/cfn-lint deploy/aws/ec2-stack.json
	@grep -Eq '^FROM python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}' Dockerfile.deploy \
	  || echo "WARN  Dockerfile.deploy base image is not pinned to a digest: run  make pin-base"

image-test:
	@if ! docker info >/dev/null 2>&1; then echo "SKIP  image build: Docker is not running here (GitHub Actions builds it on every push)"; \
	 elif [ ! -f artifacts/deployment/manifest.json ]; then echo "SKIP  image build: no bundle in artifacts/ (GitHub Actions uses a fixture bundle)"; \
	 else docker build -f Dockerfile.deploy -t job-recommender:ci . \
	   && docker run --rm --read-only --tmpfs /tmp:rw,size=512m --entrypoint python job-recommender:ci -m recommender.smoke; fi

ci: test lint image-test
	@echo "make ci: all checks finished"

pin-base:
	sh deploy/pin-base-image.sh
