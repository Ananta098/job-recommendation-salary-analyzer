# Commands for building, running, deploying and checking the project. Type `make help`.
SHELL := /bin/bash
PY312 ?= $(shell command -v python3.12 2>/dev/null || echo /opt/homebrew/opt/python@3.12/bin/python3.12)
VENV  ?= .venv
PY    := $(VENV)/bin/python
RAW   ?= datasets/HSE_dataset(D4).csv
ifndef RUN
RUN := artifacts/runs/run-$(shell date +%Y%m%d-%H%M%S)
endif

.PHONY: help venv hf-key bundle translation run \
        aws-tools aws-plan aws-deploy aws-status aws-update aws-rollback aws-versions aws-logs aws-allow aws-revoke aws-destroy \
        dev-tools test lint image-test ci pin-base

help:
	@echo "Build and run"
	@echo "  make venv          create .venv (Python 3.12) and install every requirement"
	@echo "  make hf-key        optional: save your Hugging Face key to .hf_token (git-ignored)"
	@echo "  make bundle        train, compare and export the final model to artifacts/deployment (skipped if it exists)"
	@echo "  make translation   download the English-interface models into artifacts/translation"
	@echo "  make run           start the app on http://localhost:8501"
	@echo ""
	@echo "Deploy to AWS"
	@echo "  make aws-deploy    first run creates .env.aws and stops; second run creates the server"
	@echo "  make aws-status    stack state, app address, running version and a health check"
	@echo "  make aws-update    install a new version on the running server (switched only if healthy)"
	@echo "  make aws-rollback  back to the previous version (VERSION=<name> for a specific one)"
	@echo "  make aws-versions  list the stored versions"
	@echo "  make aws-logs      recent app log lines (MINUTES=60 for more)"
	@echo "  make aws-allow CIDR=x.x.x.x/32   let another network open the app (presentation room)"
	@echo "  make aws-revoke CIDR=x.x.x.x/32  remove that access again"
	@echo "  make aws-destroy   delete the server and everything stored for it"
	@echo "  make aws-plan      dry run: bundle check, plan and cost estimate; changes nothing"
	@echo ""
	@echo "Checks"
	@echo "  make test          unit and smoke tests (no dataset, no AWS)"
	@echo "  make ci            test + lint (shellcheck, cfn-lint) + image-test (Docker build), as in GitHub Actions"
	@echo "  make pin-base      pin the Docker base image to its current digest"

# ---- Build and run ----
venv:
	@test -x "$(PY312)" || { echo "Python 3.12 not found. Install it with:  brew install python@3.12"; exit 1; }
	@test -x $(PY) || "$(PY312)" -m venv $(VENV)
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' || { echo "$(VENV) is not Python 3.12. Run  rm -rf $(VENV)  and then  make venv  again."; exit 1; }
	$(PY) -m pip install --no-cache-dir --upgrade pip
	$(PY) -m pip install --no-cache-dir -r requirements.txt
	$(PY) -m ipykernel install --user --name jobrec --display-name "Python (jobrec)"
	$(PY) -m pip check

hf-key:
	@read -r -s -p "Paste your Hugging Face token (hidden): " t; echo; \
	 [ -n "$$t" ] || { echo "Nothing entered."; exit 1; }; \
	 printf '%s\n' "$$t" > .hf_token; chmod 600 .hf_token; echo "Saved to .hf_token (git-ignored)."

# Runs the whole pipeline (clean, encode, extract skills, compare models, select, export) and copies the
# exported model bundle to artifacts/deployment. Reuses the prepared data in clean/ when it is there.
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

# The app only loads the exported bundle; it never trains a model.
run:
	@test -x $(PY) || { echo "Run  make venv  first."; exit 1; }
	@test -f artifacts/deployment/manifest.json || { echo "Build the model bundle first:  make bundle"; exit 1; }
	$(PY) -m streamlit run app.py

# ---- Deploy to AWS (one EC2 server). The only file you fill in is .env.aws ----
.env.aws:
	@cp .env.aws.example .env.aws; chmod 600 .env.aws; \
	 echo "Created .env.aws. Open it, paste the three key lines from the Learner Lab (access key, secret key, session token), save, then run this command again."; exit 1

aws-tools:
	@test -x $(PY) || { echo "Run  make venv  first."; exit 1; }
	@$(PY) -c 'import boto3' 2>/dev/null || $(PY) -m pip install --no-cache-dir -q -r requirements-deploy.txt

aws-plan: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py

aws-deploy: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --apply

aws-status: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --status

aws-update: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --update

aws-rollback: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --rollback $(VERSION)

aws-versions: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --versions

MINUTES ?= 30
aws-logs: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --logs --minutes $(MINUTES)

# Presentation access: make aws-allow CIDR=203.0.113.7/32 (or CIDR=me); undo with make aws-revoke CIDR=...
aws-allow: .env.aws aws-tools
	@test -n "$(CIDR)" || { echo "Usage: make aws-allow CIDR=203.0.113.7/32   (CIDR=me for your current IP, 0.0.0.0/0 for everyone)"; exit 1; }
	$(PY) deploy/aws/deploy.py --allow $(CIDR)

aws-revoke: .env.aws aws-tools
	@test -n "$(CIDR)" || { echo "Usage: make aws-revoke CIDR=203.0.113.7/32"; exit 1; }
	$(PY) deploy/aws/deploy.py --revoke $(CIDR)

aws-destroy: .env.aws aws-tools
	$(PY) deploy/aws/deploy.py --destroy

# ---- Checks: no raw dataset and no AWS needed ----
dev-tools:
	@test -x $(PY) || { echo "Run  make venv  first."; exit 1; }
	@$(PY) -c 'import pytest, cfnlint, shellcheck_py' 2>/dev/null || $(PY) -m pip install --no-cache-dir -q -r requirements-dev.txt

test: dev-tools
	$(PY) -m pytest -q tests

lint: dev-tools
	$(VENV)/bin/shellcheck docker/*.sh deploy/pin-base-image.sh deploy/aws/*.sh
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
