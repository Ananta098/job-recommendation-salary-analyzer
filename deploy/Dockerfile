# Base image: pin to a digest with  make pin-base  (CI warns while it is unpinned).
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
    HF_HOME=/tmp/huggingface MODEL_BUNDLE=/app/models/recommender
WORKDIR /app
COPY --chown=10001:10001 requirements/app.txt requirements/app.lock.txt ./requirements/
COPY deploy/docker/install-runtime.sh /usr/local/bin/install-runtime
RUN sh /usr/local/bin/install-runtime
COPY --chown=10001:10001 deploy/docker/entrypoint.sh /usr/local/bin/app-entrypoint
COPY --chown=10001:10001 app/ ./app/
COPY --chown=10001:10001 recommender/ ./recommender/
COPY --chown=10001:10001 .streamlit/ ./.streamlit/
COPY --chown=10001:10001 models/recommender/ ./models/recommender/
COPY --chown=10001:10001 models/translation/ ./models/translation/
RUN python -m recommender.bundle check --deep && python -m recommender.translation check --deep
USER 10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"
ENTRYPOINT ["sh", "/usr/local/bin/app-entrypoint"]
