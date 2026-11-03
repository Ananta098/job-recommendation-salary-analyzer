# Job Recommendation System

HSE D4 resume-to-vacancy recommendation for AT82.01, DSAI, Asian Institute of Technology. Authors: Nawee Pornsiriphanna (st127031) and Ananta Upadhyaya (st127094).

**Final model: M8 Hybrid + region.** Both M9 variants and Logistic Regression, Random Forest and XGBoost remain in the evaluated comparison. The conservative development-only promotion check does not justify replacing M8 on the primary grouped Recall@10 objective.

Start with **[the project handbook](docs/PROJECT_HANDBOOK.md)**. It is the single run procedure, from environment setup through a complete rebuild, app, Docker and EC2 template.

```text
Raw D4 → clean → embeddings + canonical skills → model comparison
       → selection/export → significance + regional analysis → review → app
```

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m recommender.bundle check --deep
python -m recommender.translation check --deep
streamlit run app.py
```

The original folder already contains the private serving bundle. A fresh clone needs raw D4 and a rebuild:

```sh
python -m recommender.pipeline run --run-dir artifacts/runs/run-01
```

## Layout

| Location | Responsibility |
|---|---|
| `recommender/pipeline.py` | One ordered execution and review command |
| `recommender/cleaning.py` | One D4 cleaning implementation |
| `recommender/shared.py`, `core.py` | Shared split, evidence, metrics and scoring |
| `recommender/experiment.py`, `analysis.py` | Comparison/export and paired/regional analysis |
| `recommender/serving.py`, `app.py` | Bundle validation and Streamlit app |
| `m9_skills/` | Existing extraction, dictionaries and linguistic diagnostics |
| `notebooks/01–07` | EDA and report views; no competing training runner |
| `results/final/` | Authoritative current aggregate results |
| `results/baselines/` | Explicitly historical original baseline reference |
| `artifacts/` | Ignored local bundles, runs and deployment packages |
| `deploy/aws/` | Fill-in EC2 template and deployment helper |

[Final review](docs/FINAL_REVIEW.md) · [Model verdict](docs/final_model_verdict.md)

This is a complete local project/prototype pipeline. Docker execution remains unverified because its daemon was unavailable; AWS provisioning is intentionally a template. Invitation-only labels, outcome-dependent cleaned snapshots, unreviewed skill aliases and prior inspection of the test set limit scientific claims. No production validation or future-job guarantee is claimed.

Local Docker: `docker compose -f compose.local.yaml up --build -d`. Deployment rehearsal: `docker compose -f compose.deploy.yaml up --build -d`. Both require the complete local model bundle. See the handbook before AWS deployment.

The app accepts English profiles and displays English vacancies/CSV through local OPUS-MT translation. On a fresh clone, run `python -m recommender.translation prepare` after installing dependencies and restoring/rebuilding the recommendation bundle. Translation artifacts are ignored by Git and packaged separately for Docker/EC2.
