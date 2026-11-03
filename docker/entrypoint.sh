#!/bin/sh
set -eu
python -m recommender.bundle check
python -m recommender.translation check
exec streamlit run /app/app.py --server.port=8501 --server.address=0.0.0.0 "$@"
