# Job Recommendation System

HSE D4 resume-to-vacancy recommendation for AT82.01, DSAI, Asian Institute of Technology. Authors: Nawee Pornsiriphanna (st127031) and Ananta Upadhyaya (st127094).

Given an applicant's profile, rank the historical Russian vacancies that suit them best. Salary analysis from the proposal stage is kept as supplementary work only (professor feedback); the system does not predict salary.

**Final model: M8 Hybrid + region** (TF-IDF plus multilingual Sentence-BERT, local region first). Both M9 skill-aware variants and Logistic Regression, Random Forest and XGBoost were evaluated; none passed the conservative development-only promotion check, so M8 is deployed. Start with **[the project handbook](docs/PROJECT_HANDBOOK.md)**, the single run procedure from setup to deployment.

## End-to-end flow

| Step | What happens | Where |
|---|---|---|
| 1. Data validation | Structure, IDs, label balance, missing values, duplicates, label limitations | `notebooks/01_check_data.ipynb` |
| 2. Cleaning | Deduplication, text fields, skills, English titles, evaluation set | `notebooks/02_clean_data.ipynb`, `recommender/cleaning.py` |
| 3. Baselines | Random, popularity, TF-IDF, BM25 (M0-M5) | `notebooks/03_models.ipynb` |
| 4. Semantic models | Sentence-BERT and the hybrid M8 | `notebooks/04_sentence_bert.ipynb` |
| 5. Skill-aware models | Skill extraction, M9 reranking, learned rerankers | `notebooks/06-07`, `m9_skills/` |
| 6. Evaluation and selection | Recall@K / NDCG@K, paired significance, regions; M8 selected on the development set | `notebooks/05`, `results/final/` |
| 7. Model export | Checked model bundle for serving | `python -m recommender.pipeline run` |
| 8. Application | English Streamlit app on the selected model | `app.py` |
| 9. Deployment | Docker image on one AWS EC2 server | `make aws-deploy` |

```text
Stage 1  original study (notebooks 01-04)
  Raw D4 -> validate (01) -> clean (02) -> baselines M0-M5 (03) -> Sentence-BERT M6-M8 (04)
  writes: clean/, results/baselines/, figures/

Stage 2  final system (one command + notebooks 05-07 + app)
  Raw D4 -> clean -> embeddings + canonical skills -> model comparison (M8, M9, learned rerankers)
         -> selection/export -> significance + regional analysis -> review
         -> Streamlit app -> Docker image -> live on AWS EC2 (make aws-deploy)
  writes: artifacts/, results/final/
```

Stage 1 is the study that chose the approach and is fully reproducible from the notebooks. Stage 2 repeats the same cleaning and evaluation protocol in code (`recommender/`, with an added development split), adds the skill-aware experiments, and is the only route that produces the deployable model bundle.

## Run it after cloning

Needs macOS or Linux with Python 3.12, `git` and `make`, an internet connection, and about 15 GB of free disk. The raw dataset and the trained model are not in Git (too large), so the first run rebuilds the model.

```sh
git clone https://github.com/Ananta098/job-recommendation-salary-analyzer.git
cd job-recommendation-salary-analyzer
# 1. Download HSE_dataset(D4).csv (see datasets/README.md) and put it in datasets/
make venv          # 2. Python environment with every pinned package + Jupyter kernel "Python (jobrec)"
make hf-key        # 3. optional: Hugging Face key, saved in .hf_token (ignored by Git)
make bundle        # 4. trains, compares and exports the final model to artifacts/deployment (an hour or more the first time)
make translation   # 5. downloads the English-interface models to artifacts/translation
make test          # 6. checks: bundle integrity, a sample recommendation, deploy safety
source .venv/bin/activate && streamlit run app.py   # 7. the app at http://localhost:8501
```

To follow the study itself, run `jupyter lab`, pick the **Python (jobrec)** kernel and open `notebooks/01` to `07` in order (01-04 need the dataset; 05-07 read `results/final/`). Optional: `make up` runs the app in Docker, and `make aws-deploy` puts it on AWS (see "Deploy to AWS" below).

## Quick start: run it in Docker on your computer

Needs: Docker Desktop (running), Python 3.12 (`brew install python@3.12`), about 15 GB free disk, and the raw file `HSE_dataset(D4).csv` in `datasets/` (see `datasets/README.md`). Run these from the project folder:

```sh
make hf-key        # optional: paste your Hugging Face key; saved in .hf_token (ignored by Git)
make venv          # creates .venv with Python 3.12 and installs the pinned packages
make bundle        # builds artifacts/deployment (full rebuild on a fresh clone; reuses clean/ when present; skipped if it exists)
make translation   # one-time download of the English-UI models
make check         # read-only readiness check; fix any FAIL line
make up            # builds the image, starts it, waits until healthy
```

Then open http://127.0.0.1:8501, submit a profile and confirm that recommendations appear. `make down` stops it, `make logs` shows its log, and `make local-docker` runs all the steps above in order. Re-running a step is safe.

If something fails:

| Message | Fix |
|---|---|
| `Python 3.12 not found` | `brew install python@3.12` |
| pip cannot build numpy or pandas | the venv is Python 3.13 or 3.14; `rm -rf .venv .venv-1`, then `make venv` |
| `Raw dataset missing` | copy `HSE_dataset(D4).csv` into `datasets/` |
| `Prepared input changed; do a full rebuild` | the saved inputs no longer match: `.venv/bin/python -m recommender.pipeline run --run-dir artifacts/runs/full-01` (hours on CPU), then `cp -R artifacts/runs/full-01/artifacts/deployment artifacts/deployment` |
| `No space left on device` | free about 15 GB; `pip cache purge`; in Docker Desktop run `docker system prune` |
| `Docker Desktop is not running` | open Docker Desktop and wait until it reports running |

## Run it step by step

Use the existing model bundle (app only):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m recommender.bundle check --deep
python -m recommender.translation check --deep
streamlit run app.py
```

Rebuild everything from raw D4 (see `datasets/README.md` for the file):

```sh
python -m pip install -r requirements.txt
python -m recommender.pipeline run --run-dir artifacts/runs/run-01
```

Re-run the Stage 1 study (needs `requirements.txt` and the raw D4 file):

| Step | Notebook | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `01_check_data.ipynb` | `datasets/` | `figures/` | 2 min |
| 2 | `02_clean_data.ipynb` | `datasets/`, `lookup/` | `clean/d4_clean.csv`, `results/cleaning_report.csv` | 3 min |
| 3 | `03_models.ipynb` | `clean/d4_clean.csv` | `results/baselines/metrics_*.csv`, `figures/*.png` | 5 min |
| 4 | `04_sentence_bert.ipynb` | `clean/d4_clean.csv`, `results/baselines/` | `results/baselines/*_all.csv`, `hybrid_weight.csv`, `figures/*.png` | about 16 min on a laptop CPU |
| 5-7 | `05`, `06`, `07` | `results/final/`, skill outputs, bundle | nothing by default (report views) | seconds |
| - | `archive/00_salary_analysis_supplementary.ipynb` | `datasets/ai_*.csv` | `figures/E*.png` | - |

Notebook 04 asks once for a Hugging Face key and can save it in `.hf_token` (ignored by Git). It uses an NVIDIA GPU or an Apple GPU automatically when one is available.

Notebook 01 only documents that the data is usable. Notebook 04 saves the Sentence-BERT embeddings in `clean/` (the original three caches are included), so a second run is fast, and also runs on Google Colab (steps at the top of the notebook).

## Models compared

| Model | Idea |
|---|---|
| Random | lower bound |
| M0 Popularity | most invited jobs, same for everyone |
| M1 TF-IDF title | desired title vs job title |
| M2 TF-IDF text | title + resume text vs title + job text |
| M3 BM25 text | M2 text with BM25 weighting |
| M4 Popularity + region | most invited jobs in the applicant's region, no resume |
| M5 TF-IDF text + region | M2 with jobs in the applicant's region first |
| M6 Sentence-BERT | multilingual embeddings, matches on meaning instead of words |
| M7 Sentence-BERT + region | M6 with jobs in the applicant's region first |
| M8 Hybrid + region | mix of M5 and M7 scores, weight chosen on training applicants |
| M9 (two variants) | M8 plus explicit canonical-skill evidence, with and without region |
| Logistic Regression, Random Forest, XGBoost | learned rerankers over the same candidate features |

Each model ranks all 43,756 vacancies for each of 2,806 test applicants. **Strict** scoring needs the exact posting; **grouped** accepts any posting with the same title in the same region, because many postings are identical copies.

## Results

| Model | Grouped Recall@10 (`results/baselines/`) | Grouped NDCG@10 | Final pipeline run (`results/final/`) |
|---|---|---|---|
| Random | 0.0032 | 0.0014 | - |
| M0 Popularity | 0.0449 | 0.0251 | - |
| M1 TF-IDF title | 0.1334 | 0.0680 | - |
| M2 TF-IDF text | 0.1339 | 0.0679 | - |
| M3 BM25 text | 0.1328 | 0.0712 | - |
| M4 Popularity + region | 0.1908 | 0.1162 | - |
| M5 TF-IDF text + region | 0.4558 | 0.3408 | 0.4558 |
| M6 Sentence-BERT | 0.0709 | 0.0361 | - |
| M7 Sentence-BERT + region | 0.3652 | 0.2340 | - |
| M8 Hybrid + region | **0.4660** | 0.3457 | **0.4660** |

All numbers are on the same 2,806 test applicants. The Stage 1 notebooks and the final pipeline give identical M5 and M8 scores, so the two stages agree. Region priority is the largest single gain (TF-IDF text 0.1339 → 0.4558; Sentence-BERT 0.0709 → 0.3652), and the hybrid M8 adds a further small step over M5 (+0.010, Holm-adjusted p = 0.048 in `results/final/significance.csv`). Strict Recall@10 (exact posting) for M8 is 0.2749. The skill-aware variants, the learned rerankers, the region control (M8 without region), paired significance tests and regional counts are in `results/final/`, `docs/final_model_verdict.md` and notebooks 05-07.

## Professor feedback: where each point is done

| Feedback | Where |
|---|---|
| Focus on job recommendation; salary supplementary only | Recommender and app only; salary analysis kept in `notebooks/archive/` with its saved outputs (its data files are not used by the model; source in `datasets/README.md`) |
| Validate D4 (structure, IDs, labels, balance, categories, missing values, duplicates, label limitations) | Notebook 01, cleaning in notebook 02 |
| Baseline TF-IDF/BM25, Sentence-BERT, skill-aware improvement, Recall@K and NDCG@K on a train/test split | Notebooks 03-04 (M0-M8); `m9_skills/` and notebooks 06-07 (M9, learned rerankers); metrics at K = 5, 10, 20 |
| Research question: does skill-aware semantic matching help when terminology differs | Overall: `results/final/significance.csv`, `docs/final_model_verdict.md` (no challenger beat M8). Terminology subgroups: notebook 05, last section, and `results/final/terminology_subgroups.csv`: no gain for applicants whose CV shares few words with their invited vacancies, because few of them have recognisable skills |

## Deploy to AWS

The only file you fill in is `.env.aws` (ignored by Git, never copied into the image or the package):

```sh
make aws-deploy     # first run creates .env.aws and stops: paste the three Learner Lab key lines into it
make aws-deploy     # creates the server and prints the app address when it answers (15-20 minutes)
```

| Command | What it does |
|---|---|
| `make aws-plan` | Dry run: verifies the bundle, prints the plan, version and estimated hourly cost. Changes nothing. |
| `make aws-update` | Uploads a new version and installs it on the running server. The new image is built (or reused), smoke-tested and health-checked next to the live app; the switch only happens if it is healthy, and a failed switch restores the previous version. |
| `make aws-rollback` | Returns to the previous version and confirms the health URL answers. `make aws-rollback VERSION=<name>` picks a specific one. |
| `make aws-versions` | Versions stored in S3 (`*` running, `<` previous). |
| `make aws-status` | Server state, current address, running version and a live health check. |
| `make aws-logs` | Recent app log lines (CloudWatch, or straight from the server). `MINUTES=60` for more. |
| `make aws-allow CIDR=…` | Lets another network open the app without redeploying (for example the presentation room's address `/32`, `CIDR=me` for your current IP, `0.0.0.0/0` for everyone). `make aws-revoke CIDR=…` removes it. |
| `make aws-destroy` | Deletes the server and stored packages. |

Versions are named `<UTC time>-<git commit>`, images are tagged with that name, and the running version shows in the app footer and sidebar. App and startup logs go to CloudWatch (`LOG_RETENTION_DAYS`, default 14); if the account does not allow it, this is skipped with a message and `make aws-logs` reads the server directly. A timer on the server restarts the app after three failed health checks in a row. The app is plain HTTP and only admits your current IP (`ALLOWED_CIDR`); handbook section 5 describes the optional HTTPS route for showing it to others. Learner Lab stops the server at the end of a session: `make aws-update`/`aws-rollback` start it again, and `make aws-status` shows the new address.

**App is down:** `make aws-status`, then `make aws-logs`. If the server is stopped, start the lab and run `make aws-status` again (or any update/rollback, which starts it). If health stays "NOT ANSWERING", run `make aws-rollback` to the last version that worked.
**Bad release:** `make aws-rollback` (previous version) or `make aws-versions` and `make aws-rollback VERSION=<name>`.

Local checks before deploying (optional): `make check`, then `make up`. The Docker image and the EC2 server need the complete private bundle in `artifacts/deployment/` and the translation assets in `artifacts/translation/`; these are excluded from Git.

## Tests and CI

```sh
make test      # pytest: bundle and translation integrity (incl. tampered files), serving smoke test, deploy allowlist and AWS command flows
make ci        # make test + shellcheck + cfn-lint + Docker image build and smoke test (skipped when Docker is not running)
make pin-base  # pin Dockerfile.deploy to the current digest of python:3.12-slim-bookworm (commit the change)
```

The tests need neither the raw dataset nor AWS: they build a tiny fixture bundle and use an in-memory stand-in for AWS. Tests that use the real bundle run only on a computer that has `artifacts/`. GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on every push and pull request and builds `Dockerfile.deploy` with the fixture bundle. CI never deploys and holds no AWS secrets, because Learner Lab keys expire; deployment stays a manual `make aws-deploy` / `make aws-update`.

## Layout

| Location | Responsibility |
|---|---|
| `notebooks/01-04` | Stage 1 study: validation, cleaning, baselines, Sentence-BERT and hybrid (with saved outputs) |
| `notebooks/05-07` | Stage 2 report views: significance and regions, skills, rerankers and final model |
| `notebooks/archive/` | Supplementary salary analysis from the proposal stage |
| `recommender/pipeline.py` | One ordered execution and review command |
| `recommender/cleaning.py` | D4 cleaning as a callable (same logic as notebook 02) |
| `recommender/shared.py`, `core.py` | Shared split, evidence, metrics and scoring |
| `recommender/experiment.py`, `analysis.py` | Comparison/export and paired/regional analysis |
| `recommender/serving.py`, `app.py` | Bundle validation and Streamlit app |
| `m9_skills/` | Skill extraction, dictionaries and linguistic diagnostics |
| `datasets/` | Raw D4 goes here (not in Git); small salary files tracked; see `datasets/README.md` |
| `lookup/` | `job_titles_ru_en.csv`, 550 job titles translated to English |
| `clean/` | Prepared data and Sentence-BERT embedding caches (ignored by Git) |
| `results/final/` | Authoritative current aggregate results |
| `results/baselines/` | Stage 1 results, kept as the historical reference |
| `figures/` | Charts from notebooks 01, 03, 04 |
| `artifacts/` | Ignored local bundles, runs and deployment packages |
| `deploy/` | `preflight.sh`, `pin-base-image.sh`; `aws/`: EC2 template, `deploy.py`, on-server release, health-check and CloudWatch agent files |
| `tests/` | pytest suite and fixture bundle builder (`python -m tests.fixtures --out DIR`) |
| `.github/workflows/ci.yml` | CI: lint, tests, image build; never deploys |

[Final review](docs/FINAL_REVIEW.md) · [Model verdict](docs/final_model_verdict.md)

## Limits

This is a complete local project/prototype pipeline. Docker image builds and AWS provisioning have not been run: the Docker daemon was unavailable during the handoff, and AWS is intentionally a template. Invitation-only labels, outcome-dependent cleaned snapshots, unreviewed skill aliases and prior inspection of the test set limit scientific claims. No production validation or future-job guarantee is claimed.

The raw file is separated by `|`, not commas. Gender, age and birthday are never loaded. The models read the Russian text; English titles are only for reading examples.

The app accepts English profiles and displays English vacancies/CSV through local OPUS-MT translation. On a fresh clone, run `python -m recommender.translation prepare` after installing dependencies and restoring/rebuilding the recommendation bundle. Translation artifacts are ignored by Git and packaged separately for Docker/EC2.

Docker: `make up` (or `docker compose --env-file .env.deploy -f compose.deploy.yaml up --build -d`). It needs the complete local model bundle. See the handbook before AWS deployment.
