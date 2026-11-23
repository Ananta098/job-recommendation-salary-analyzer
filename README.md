# Job Recommendation System

Course project for AT82.01, Asian Institute of Technology (AIT).
Authors: Nawee Pornsiriphanna (st127031) and Ananta Upadhyaya (st127094).

Given an applicant's profile (desired job title, CV text, skills, region), the system ranks the vacancies of the HSE D4 dataset and returns the ones that suit the applicant best. We built it from the raw data to a live website: data checks, cleaning, a series of ranking models, a statistical comparison, a Streamlit app, and a deployment on one AWS EC2 server.

**Final model: M8 Hybrid + region.** It mixes a TF-IDF text score with a multilingual Sentence-BERT score and puts vacancies in the applicant's own region first. On 2,806 test applicants it reaches a grouped Recall@10 of 0.4660 (`results/final/metrics_grouped.csv`). Skill-aware reranking (M9) and three learned rerankers were also tried; none of them beat M8, so M8 is the model in the app.

## 1. Problem and dataset

**Dataset.** HSE D4, a Russian resume and vacancy dataset from an HSE hackathon: <https://www.kaggle.com/datasets/darysha/hse-hackathon>. It is not in this repository (about 1 GB); see `data/README.md` for where to put it.

- **What a row is:** one application, that is one resume paired with one vacancy, with the texts of both and whether the employer invited the applicant.
- **Size** (`results/figures/D4_summary.csv`): 150,493 rows, 65,714 resumes, 43,756 vacancies. After removing duplicates 93,755 rows remain (`results/cleaning_report.csv`).
- **Labels:** an invitation is our only sign that a vacancy was relevant. The invitation rate is 13.9%, and 14,033 applicants have at least one invitation; these are the applicants we can evaluate on.
- **Why we rank the whole catalog:** an applicant has only 2.29 applications on average and a median of 1 (`results/figures/D4_summary.csv`). Re-ordering one or two applied jobs says nothing, so each model ranks all 43,756 vacancies for each applicant and we check whether the invited ones come out near the top.
- **Privacy:** gender, age and birthday are never loaded.

**Scope change.** The proposal also included salary analysis. Following the professor's feedback the project focuses on job recommendation only. The salary work is kept as supplementary material in `notebooks/archive/` and is not used by the model or the app.

## 2. End-to-end pipeline

| # | Step | Notebook | Code | Output |
|---|---|---|---|---|
| 1 | Check the raw data: structure, IDs, label balance, missing values, duplicates | `01_check_data.ipynb` | - | `results/figures/D4_*.png`, `results/figures/D4_summary.csv` |
| 2 | Clean: remove duplicates, build CV and vacancy texts, skills, English titles, evaluation set | `02_clean_data.ipynb` | `recommender/cleaning.py` | `data/processed/d4_clean.csv`, `results/cleaning_report.csv` |
| 3 | Baseline models M0-M5: popularity, TF-IDF, BM25, region | `03_models.ipynb` | `recommender/core.py`, `shared.py` | `results/baselines/` |
| 4 | Sentence-BERT models M6-M7 and the hybrid M8 | `04_sentence_bert.ipynb` | `recommender/core.py` | `data/processed/sbert_*.npy`, `results/baselines/*_all.csv` |
| 5 | Skill extraction and normalisation for M9 | `06_pull_and_normalize_skills.ipynb` | `recommender/skills/` | `recommender/skills/outputs/`, `recommender/skills/validation/` |
| 6 | Compare M8, M9 and the learned rerankers; select on the development set; export the model | `07_rerankers_and_final_model.ipynb` | `recommender/experiment.py` | `results/final/metrics_*.csv`, `selection.json`, `models/recommender/` |
| 7 | Significance tests, regions, research question | `05_significance_and_regions.ipynb` | `recommender/analysis.py`, `terminology.py` | `results/final/significance.csv`, `terminology_subgroups.csv` |
| 8 | Web app with an English interface | - | `app/streamlit_app.py`, `recommender/serving.py`, `translation.py`, `profile_form.py` | the app on port 8501 |
| 9 | Deployment on AWS EC2 | - | `deploy/Dockerfile`, `deploy/aws/`, `Makefile` | a public address |

Steps 2 and 4 to 7 are also run in order by one command, `python -m recommender.pipeline run` (this is what `make bundle` calls). The notebooks show the same work with explanations and saved outputs.

## 3. Methodology

**Split.** Applicants, not rows, are split, so no applicant appears in two parts: 10,227 for fitting, 1,000 for development and 2,806 for testing, seed 42 (`results/final/selection.json`).

**Models.**

| Model | Idea |
|---|---|
| Random | lower bound |
| M0 Popularity | the most invited jobs, the same for everyone |
| M1 TF-IDF title | desired title against job title |
| M2 TF-IDF text | title + CV text against title + job text |
| M3 BM25 text | the M2 texts with BM25 weighting |
| M4 Popularity + region | the most invited jobs in the applicant's region |
| M5 TF-IDF text + region | M2 with jobs in the applicant's region first |
| M6 Sentence-BERT | multilingual sentence embeddings, matches on meaning |
| M7 Sentence-BERT + region | M6 with jobs in the applicant's region first |
| **M8 Hybrid + region** | equal mix of the M5 and M7 scores (weight 0.5, `results/baselines/hybrid_weight.csv`) |
| M9 without region / M9 + region | M8's scores plus a bonus for skills that the CV and the vacancy share |
| Logistic Regression, Random Forest, XGBoost | learned rerankers over the same candidates, using ten features such as the TF-IDF and semantic scores, skill coverage and same region |
| M8 without region | control: M8 with the region rule switched off |

**Metrics.** Recall@K, Precision@K and NDCG@K at K = 5, 10 and 20. The main metric is grouped Recall@10.

- **Strict:** the exact invited posting must be in the top K.
- **Grouped:** any posting with the same title in the same region counts, because many postings are copies of each other.

**Model selection.** The choice is made on the development set only. A challenger replaces M8 only if the lower bound of its paired bootstrap interval for the gain in grouped Recall@10 is above zero after correcting for the five comparisons (10,000 resamples, `results/final/selection.json`). This rule is conservative on purpose: we keep the simpler model unless there is clear evidence. No challenger passed (`results/final/development_promotion.csv`).

**Significance on the test set.** One paired permutation test per comparison with Holm correction, plus paired bootstrap 95% intervals (`results/final/analysis_protocol.json`).

## 4. Results

All numbers are for the same 2,806 test applicants and 43,756 vacancies.

| Model | Grouped Recall@10 | Grouped NDCG@10 | Source |
|---|---:|---:|---|
| Random | 0.0032 | 0.0014 | baselines |
| M0 Popularity | 0.0449 | 0.0251 | baselines |
| M1 TF-IDF title | 0.1334 | 0.0680 | baselines |
| M2 TF-IDF text | 0.1339 | 0.0679 | baselines |
| M3 BM25 text | 0.1328 | 0.0712 | baselines |
| M4 Popularity + region | 0.1908 | 0.1162 | baselines |
| M5 TF-IDF text + region | 0.4558 | 0.3408 | baselines and final |
| M6 Sentence-BERT | 0.0709 | 0.0361 | baselines |
| M7 Sentence-BERT + region | 0.3652 | 0.2340 | baselines |
| **M8 Hybrid + region** | **0.4660** | 0.3457 | baselines and final |
| M8 without region (control) | 0.1334 | 0.0674 | final |
| M9 without region | 0.1312 | 0.0668 | final |
| M9 + region | 0.4641 | 0.3441 | final |
| Logistic Regression | 0.4656 | 0.3512 | final |
| Random Forest | 0.4650 | 0.3506 | final |
| XGBoost | 0.4660 | 0.3494 | final |

Sources: "baselines" is `results/baselines/metrics_grouped_all.csv` (notebooks 03 and 04) and "final" is `results/final/metrics_grouped.csv` (the pipeline). M5 and M8 appear in both files with identical scores. XGBoost is 0.46597 and M8 is 0.46603; they round to the same value.

Strict Recall@10 for M8 is 0.2749 (`results/final/metrics_strict.csv`).

## 5. Findings

1. **Region priority is the largest single gain.** Putting the applicant's region first lifts TF-IDF text from 0.1339 to 0.4558 and Sentence-BERT from 0.0709 to 0.3652. The control confirms it: M8 without region drops to 0.1334, a difference of −0.333 (`results/final/significance.csv`, family "region effect (control)"). 82.6% of applications are in the applicant's own region (`results/cleaning_report.csv`).
2. **The hybrid adds a small but significant gain.** M8 is 0.010 above M5 in grouped Recall@10, with a 95% interval of 0.003 to 0.018 and a Holm-adjusted p of 0.047 (`results/final/significance.csv`).
3. **Skill-aware reranking and learned rerankers do not beat M8.** M9 + region is 0.002 below M8 and the interval includes zero. The three learned rerankers are within 0.001 of M8 on grouped Recall@10. Logistic Regression is better on strict Recall@10 (+0.009, Holm-adjusted p 0.040), but it did not pass the development rule on the main metric, so it was not selected.
4. **Research question: does skill-aware matching help when the CV and the vacancy use different words? No.** For the third of applicants with the lowest vocabulary overlap, M8 scores 0.1686 and M9 + region 0.1676 (`results/final/terminology_subgroups.csv`). The reason is coverage: only 12.3% of these applicants have a skill that our dictionary recognises, so the skill score rarely changes the ranking. Overlap itself matters a lot: M8 scores 0.1686 in the low-overlap third and 0.7062 in the high-overlap third.
5. **The skill term adds nothing with or without region.** Compared with its matching control (M8 without region), M9 without region differs by −0.002, with an interval of −0.006 to 0.002 (`results/final/significance.csv`, family "region-matched control").

More detail on each model is in `docs/final_model_verdict.md`.

## 6. Limitations

- **Labels are invitations.** A vacancy without an invitation is not proven irrelevant, and we can only evaluate applicants who received at least one invitation.
- **The test set was looked at more than once.** The same test applicants were used across project iterations, so the results are exploratory. There is no leakage between training and test applicants.
- **Small skill dictionary.** Dictionary v2 has 47 skills and has not been reviewed by a human expert. Missing skill evidence means "unknown", not "no skill".
- **No temporal split.** The data is not split by date, so we do not claim the model works on future vacancies.
- **The English interface is not evaluated.** The published scores use the original Russian profiles. English profiles are machine-translated before ranking, and translation errors can change the results.
- **Demo-grade deployment.** Plain HTTP, one server, and AWS Academy Learner Lab sessions that stop the server when the lab ends.

## 7. How to run

**Prerequisites:** macOS or Linux, Python 3.12, `git`, `make`, an internet connection and about 15 GB of free disk.

1. Clone the repository.
   ```sh
   git clone https://github.com/Ananta098/job-recommendation-salary-analyzer.git
   cd job-recommendation-salary-analyzer
   ```
2. Download the dataset from the Kaggle link above and save it as `data/raw/HSE_dataset(D4).csv`.
3. Create the environment.
   ```sh
   make venv
   ```
4. Build the model. This cleans the data, encodes all vacancies, compares the models and exports the final one to `models/recommender/`. On a fresh clone it takes an hour or more on a laptop CPU; later runs are skipped if the bundle exists.
   ```sh
   make bundle
   ```
5. Download the two translation models for the English interface.
   ```sh
   make translation
   ```
6. Start the app and open <http://localhost:8501>.
   ```sh
   make run
   ```

The model files and the dataset are too large for Git, which is why steps 2 to 5 are needed once.

**Reading the notebooks.** Start Jupyter from the environment, choose the kernel "Python (jobrec)" and open them in this order: 01 (data check), 02 (cleaning), 03 (baselines), 04 (Sentence-BERT and hybrid), 06 (skills), 07 (rerankers and final model), 05 (significance, regions and the research question). All notebooks keep their saved outputs, so they can also be read on GitHub without running anything. Notebooks 01 to 04 need the raw dataset to re-run; 05 to 07 read the files in `results/final/`. Notebook 04 can use a Hugging Face key (`make hf-key`, optional).

## 8. Deploy to AWS

The app runs as a Docker container on one EC2 server (t3.large), created with CloudFormation. Build the bundle first (section 7, steps 2 to 5), because the model files are not in Git and are uploaded from your computer.

1. Start the AWS Academy Learner Lab and wait for the green dot.
2. Run the deploy command once. It creates `.env.aws` and stops.
   ```sh
   make aws-deploy
   ```
3. Open `.env.aws` and paste the three key lines from the lab page (AWS Details > AWS CLI): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`. This file is ignored by Git and is never uploaded.
4. Run the command again. It uploads the app, creates the server and prints the address after 15 to 20 minutes.
   ```sh
   make aws-deploy
   ```
5. Open the printed address (`http://<address>:8501`) and run a search.

Useful commands afterwards:

| Command | What it does |
|---|---|
| `make aws-status` | Stack state, app address, running version and a health check |
| `make aws-allow CIDR=x.x.x.x/32` | Lets another network open the app, for example the presentation room. `make aws-revoke CIDR=...` removes it |
| `make aws-destroy` | Deletes the server and everything stored for it |
| `make aws-plan` | Dry run: checks the bundle and prints the plan and cost estimate. Changes nothing |

Notes:

- **Access.** By default only your current IP address can open the app.
- **Keys expire.** Learner Lab keys last for one lab session. For a new session, paste the three new lines into `.env.aws`.
- **Cost.** About $0.09 per hour while the server runs. Run `make aws-destroy` when you are finished.
- **Address changes.** The lab stops the server at the end of a session. `make aws-status` shows the new address after a restart.

## 9. Updating and rolling back

`make aws-update` uploads the current code and model as a new version and installs it on the running server. The new version is built and health-checked next to the live app, and the switch happens only if it is healthy. `make aws-rollback` returns to the previous version (`VERSION=<name>` picks a specific one), `make aws-versions` lists the stored versions, and `make aws-logs` shows recent app log lines.

## 10. Tests and CI

```sh
make test   # pytest
make ci     # tests + shellcheck + cfn-lint + Docker image build
```

- `make test` checks the model bundle and translation files (including tampered files), the serving code on a sample profile, the profile form, the list of files that may be uploaded to AWS (no secrets, no data), and the deploy, update and rollback flows against a fake AWS.
- `make ci` adds shell-script and CloudFormation linting and builds the Docker image with a smoke test (skipped when Docker is not running).
- The tests need neither the dataset nor AWS: they build a tiny fixture bundle.
- GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on every push and pull request. CI never deploys and holds no AWS keys; deployment is always a manual `make aws-deploy` or `make aws-update`.

## 11. Project structure

The folders follow the order of the work: data, notebooks, pipeline code, results, trained models, app, deployment.

```text
.
├── data/                    everything the models read
│   ├── raw/                 HSE D4 and the salary files go here (not in Git; see data/README.md)
│   ├── lookup/              550 job titles translated to English
│   └── processed/           cleaned data and Sentence-BERT embeddings (generated, not in Git)
├── notebooks/               01-07: the study step by step, with saved outputs
│   └── archive/             supplementary salary analysis (not used by the model)
├── recommender/             Python package: the whole pipeline and the serving code
│   ├── cleaning.py          data cleaning (same logic as notebook 02)
│   ├── shared.py, core.py   split, metrics, scoring, M5-M8
│   ├── experiment.py        model comparison, selection and export
│   ├── analysis.py          significance tests and regional analysis
│   ├── terminology.py       research-question subgroup analysis
│   ├── pipeline.py          runs every step in order (make bundle)
│   ├── bundle.py            checks the exported model bundle
│   ├── serving.py           loads the bundle and ranks vacancies
│   ├── translation.py       English <-> Russian adapter for the app
│   ├── profile_form.py      structured profile form
│   ├── smoke.py             quick "does it rank" test used in the image
│   └── skills/              M9 skill dictionaries, extraction and validation
├── results/
│   ├── baselines/           results of notebooks 03-04 (M0-M8)
│   ├── final/               results of the final pipeline run
│   ├── figures/             charts from notebooks 01, 03 and 04
│   └── cleaning_report.csv  what each cleaning step removed
├── models/                  trained model bundle and translation models (generated by make bundle / make translation, not in Git)
├── app/                     Streamlit web app (streamlit_app.py)
├── deploy/
│   ├── Dockerfile           the container image the server runs
│   ├── docker/              start-up and install scripts used inside the image
│   ├── aws/                 CloudFormation template, deploy.py and on-server scripts
│   └── pin-base-image.sh    pins the Docker base image
├── docs/                    final_model_verdict.md: the model decision in detail
├── tests/                   pytest suite and fixture bundle
├── requirements/            pinned packages: app, app.lock, experiment, deploy, dev
├── requirements.txt         installs everything for the notebooks and the pipeline (make venv)
├── Makefile                 all commands (make help)
└── README.md
```

Hidden files at the top level are tool settings: `.github/workflows/ci.yml` (GitHub Actions), `.streamlit/config.toml` (app theme and port), `.dockerignore`, `.gitignore` and `.env.aws.example` (template for your AWS keys).

## 12. References

- **Dataset:** HSE hackathon resume and vacancy data (D4), <https://www.kaggle.com/datasets/darysha/hse-hackathon>.
- **Sentence-BERT model:** `paraphrase-multilingual-MiniLM-L12-v2`, <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>. Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019.
- **Translation models:** Helsinki-NLP OPUS-MT, <https://huggingface.co/Helsinki-NLP/opus-mt-en-ru> and <https://huggingface.co/Helsinki-NLP/opus-mt-ru-en>. Tiedemann and Thottingal, "OPUS-MT: Building open translation services for the World", EAMT 2020.

## 13. Use of AI Tools

We used Claude (Anthropic) as a writing and editing assistant. It helped
draft parts of this README,such as compilation and helped
restructure and format the final report and presentation. All code, data
processing, experiments and results are our own work. We reviewed and
tested every command and claim in this repository before submission.
