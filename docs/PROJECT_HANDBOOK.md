# Project handbook and final report

Job Recommendation System · final handoff · 3 November 2026

## 1. What this project does and the final decision

The project ranks historical Russian vacancies for an applicant using job title, work history, education, skills and optional region. It displays listed salaries; **it does not train a salary prediction model**. The historical repository name includes “Salary Analyzer”, but the completed pipeline is a job recommendation prototype.

**Use M8 Hybrid + region for deployment.** The exported `metadata.json` selects M8. Both M9 variants and the three learned rerankers remain visible in the results and notebook 07. No model should be presented as deliberately weakened to make M8 win: the choice follows the recorded development comparison and retention rule.

The evaluated catalog has 43,756 vacancies. Applicant splits are 10,227 training, 1,000 development and 2,806 test. Applicant identities are split apart. Primary evaluation is grouped Recall@10: relevance matches an invited vacancy's normalized title and region. Strict Recall@10 requires the exact vacancy. Recall is averaged per applicant; these values are not hiring success probabilities.

| Model | Dev grouped R@10 | Test grouped R@10 | Test strict R@10 | Interpretation and limitation |
|---|---:|---:|---:|---|
| M8 hybrid + region | 0.4425 | 0.466031 | 0.274941 | Retained final; simple and strong on primary objective. Geographic priority can miss relocation opportunities; semantic input is truncated to 400 characters. |
| M9 without region | 0.1135 | 0.131213 | 0.044449 | Global skill-aware alternative performs poorly in this region-sensitive dataset. This difference reflects both geography and skill scoring, not a clean isolated skill effect. |
| M9 + region | 0.4445 | 0.464071 | 0.273634 | Fairer geographic comparison; tiny development gain does not pass uncertainty gate. Dictionary coverage and alias quality limit skill evidence. |
| Logistic Regression | 0.4345 | 0.465615 | 0.284254 | Interpretable learned feature weights; improves strict secondary metric, but not primary development objective. Weak negatives limit label quality. |
| Random Forest | 0.4400 | 0.464962 | 0.280167 | Captures interactions, with greater size and overfitting risk; no primary promotion justification. |
| XGBoost | 0.4395 | 0.465972 | 0.284284 | Strong strict secondary result, extra complexity and tuning; no primary development promotion justification. |
| M5 comparator | — | 0.455803 | 0.266221 | Retained baseline reference, below M8 on primary metric. |

These are the current `results/final/` figures, not historical notebook 03 scores. M8 is retained unless a challenger has a positive lower bound for paired development improvement using 10,000 bootstrap resamples, family correction across five challengers (99% individual intervals). No challenger qualifies. This conservative policy was added after exploration and was **not preregistered**. Test figures describe performance, rather than choosing the deployed model. Prior test inspection still limits claims of a pristine holdout. Paired permutation tests, bootstrap uncertainty, Holm correction and regional sample counts are reported in notebook 05; they are complementary analysis rather than extra training stages.

### What is used and why

| Component | Role and reason |
|---|---|
| Python 3.12, pandas, NumPy, SciPy | Supported runtime; data preparation and efficient sparse/dense scoring. |
| TF-IDF + scikit-learn | Exact vocabulary/term matching; useful for occupational keywords. |
| Frozen multilingual MiniLM / Sentence Transformers / CPU PyTorch | Semantic similarity for Russian profiles and vacancies; no encoder fine-tuning. Pinned revision avoids changing embeddings between runs. |
| M8 hybrid | Equal-weight lexical/semantic standardized scores combine literal and semantic matching. |
| Region priority | Prioritizes matching-region candidates, reflecting observed invitation geography. |
| Canonical skill dictionary v2 (47 entries) | Auditable skill aliases, negation/evidence handling and M9 experiments; limited scope and human review pending. |
| Logistic Regression, Random Forest, XGBoost | Compare linear and nonlinear reranking of the same candidate features. |
| Streamlit | Profile form, ranked vacancies and CSV download without a separate frontend service. |
| Docker | Repeatable CPU runtime with pinned dependencies and a local encoder; no model download at serving time. |
| EC2 + CloudFormation + S3 + SSM | Explicit infrastructure template, private package transfer and instance administration without opening SSH. |

M8 computes `0.5 × z(TF-IDF cosine) + 0.5 × z(semantic cosine)`. Standardization uses the full vacancy catalog for each applicant, before candidate/region prioritization. The shared core retrieves up to 300 candidates; M9 adds canonical skill evidence, and learned alternatives rerank shared features. The app uses the exported specification, vectorizer, catalog, matrices and local encoder. It does not refit on startup. M8 does not use a learned skill weight even though the common bundle includes skill evidence for the shared serving interface.

### Project flow and ownership

```text
Raw D4 → cleaning → pinned embeddings + canonical skills
       → development/model comparison → selection + bundle export
       → paired significance + regional counts → consistency review
       → Streamlit → local Docker → deployment Docker → EC2 template
```

`recommender.pipeline` is the single computational runner for the deployable model. `cleaning.py` prepares inputs; `shared.py` and `core.py` own common scoring/metrics; `experiment.py` compares/exports; `analysis.py` creates statistical and regional reports; `bundle.py` resolves/checks artifacts; `serving.py` performs inference. Notebooks 01–04 reproduce the original Stage 1 study (validation, cleaning, M0–M8) and write only historical outputs to `clean/`, `results/baselines/` and `figures/`; notebooks 05–07 are report views. None of them is required to build or serve the model bundle. `results/final/` is authoritative; `results/baselines/` is historical.

## 2. Fixing “Model bundle missing” and running the existing app

The default bundle path is now anchored to the project directory, so launching Streamlit from another folder works. Relative `MODEL_BUNDLE` values also resolve against the project directory; absolute paths are accepted. The error now prints the resolved path and recovery choices. A startup preflight checks required files; the serving loader verifies every manifest checksum.

The original project contains a local bundle at `artifacts/deployment/`. A Git clone or source ZIP excludes it because models/private catalog artifacts are ignored. Source code alone cannot serve recommendations. Restore the **entire trusted bundle**, including its `encoder/` directory, or follow the rebuild route below. Do not fabricate a manifest or copy only the weights. Joblib artifacts should only come from your own trusted export.

```sh
cd /Users/anantaupadhyaya/Downloads/Job-Recommendation-Salary-Analyzer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m recommender.bundle check --deep
python -m recommender.translation check --deep
streamlit run app.py
```

Open `http://localhost:8501`. Enter English profile text. The local adapter translates it to Russian for M8; results are translated to English. If your bundle is elsewhere:

```sh
MODEL_BUNDLE=/absolute/path/to/complete/bundle streamlit run app.py
```

For notebooks and rebuilds install `requirements.txt` instead. No retraining is needed to use the existing bundle. If an error remains, run the check command and use the exact missing path it prints. Do not treat a missing bundle as a reason to change the chosen model.

## 2. Rebuild from raw data

This is the route for a new clone, where private data and model artifacts are absent.

1. Complete the environment setup above.
2. Put the original pipe-delimited D4 file at `datasets/HSE_dataset(D4).csv`.
3. Run one command from the repository root:

```sh
python -m recommender.pipeline run --run-dir artifacts/runs/run-01
```

The command cleans D4, resolves the pinned multilingual MiniLM model, creates catalog/development/test embeddings, extracts dictionary-v2 skills, fits and compares the models, selects/exports the serving model, computes significance/region summaries and reviews the outputs. The first run needs Internet access to obtain the encoder if it is not cached. Encoding is the slowest stage.

The frozen encoder revision is `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. To use an existing local snapshot:

```sh
python -m recommender.pipeline run --run-dir artifacts/runs/run-02 --encoder /absolute/path/to/encoder/snapshot
```

Choose a **new** run directory each time. The pipeline refuses to overwrite a nonempty run. If a step fails, `pipeline_status.json` records the step and error; resolve the issue and choose a new run name.

A completed run contains:

```text
artifacts/runs/run-01/
  clean/                     prepared data and evaluation embeddings
  m9_skills/outputs/         canonical skill evidence
  results/final/             model metrics, selection, significance and regions
  artifacts/deployment/      selected model bundle and local encoder
  pipeline_status.json       ordered execution status
  final_review.json          consistency-review result
```

Serve that run without changing the reference app/bundle:

```sh
MODEL_BUNDLE="$PWD/artifacts/runs/run-01/artifacts/deployment" streamlit run app.py
```

Review or analyze that specific run:

```sh
python -m recommender.pipeline review --root artifacts/runs/run-01
python -m recommender.analysis --root artifacts/runs/run-01 --results artifacts/runs/run-01/results/final
```

The second command is needed only if you deliberately regenerate the analysis; the complete pipeline already performs it.

### Faster rerun using the existing prepared inputs

In this original folder, a new comparison can reuse the prepared inputs:

```sh
python -m recommender.pipeline run --run-dir artifacts/runs/run-03 --reuse-prepared
```

This option verifies every recorded prepared-input hash and the raw D4 hash before copying the inputs. It needs, in this folder: `clean/d4_clean.csv`, the three `clean/sbert_*.npy` caches (the original ones are included and match the recorded hashes), `m9_skills/dictionary.v2.json`, and `m9_skills/outputs/v2_validation_run/` with `cv_skills.jsonl`, `vacancy_skills.jsonl` and `coverage.json`. The d4_clean.csv and the two skill JSONL files are generated, not tracked, so on a machine that lacks them use the full rebuild above. It skips cleaning/encoding/extraction already completed for those exact inputs. If any hash differs, it stops; use a full rebuild. It still fits/evaluates models, exports an independent bundle, computes analysis and runs the final review.

## 3. Read the notebooks

Start Jupyter after the pipeline finishes:

```sh
jupyter lab
```

| Notebook | Purpose |
|---|---|
| 01 | Optional raw-data EDA; unique data-quality/occupation diagnostics |
| 02 | Original step-by-step cleaning procedure with saved outputs; the same logic runs as `recommender/cleaning.py` |
| 03 | Original M0–M5 baselines (popularity, TF-IDF, BM25, region priority), historical; writes `results/baselines/` |
| 04 | Original Sentence-BERT M6–M7 and hybrid M8 with weight selection, historical; writes `results/baselines/` |
| 05 | Current paired significance and regional sample/performance summaries |
| 06 | Current skill evidence and retained linguistic diagnostics |
| 07 | Current M9/reranker results, promotion decision and exported bundle |
| archive/00 | Supplementary salary analysis from the proposal stage (needs `datasets/ai_job_dataset.csv` and `datasets/ai_job_salary_dataset_10k.csv`, not in Git; see `datasets/README.md` for the source). Saved outputs are readable without them. |

Read 01–07 in order for presentation. Notebooks 01–04 can be re-run on the raw D4 file in that order (about 30 minutes in total on a laptop CPU; see the README table). They never touch `results/final/` or `artifacts/`. Re-running notebook 02 rewrites `clean/d4_clean.csv`; if its bytes differ from the recorded run, `--reuse-prepared` will refuse the file and you should use a full rebuild. Notebook 04 reuses the Sentence-BERT caches in `clean/` only when their row counts match the clean file. Training the deployable model needs no special notebook order: the CLI handles it. Reports 05 and 07 use `results/final/` by default. To inspect a fresh run instead:

```sh
RESULTS_DIR="$PWD/artifacts/runs/run-01/results/final" \
MODEL_BUNDLE="$PWD/artifacts/runs/run-01/artifacts/deployment" jupyter lab
```

Notebook 06 documents the frozen extraction/linguistic diagnostics. Per-run extraction summaries are in that run's `m9_skills/outputs/v2_validation_run/coverage.json`.

## 4. Docker

The shortest route is `make local-docker` (see the README quick start); the commands below are what it runs. Run `sh deploy/preflight.sh` first: it verifies the bundle and translation assets (full checksums), the files the image copies, the Compose file and your AWS settings, and prints what to fix. Then start Docker Desktop. The image runs as UID 10001, use CPU PyTorch, pinned serving requirements, a read-only filesystem and a writable temporary directory. `docker/install-runtime.sh` and `docker/entrypoint.sh` keep installation/startup procedures shared. The entrypoint rejects a missing bundle before starting Streamlit. The loader then verifies checksums.

### Deployment rehearsal: immutable source/model image (what `make up` runs)

`Dockerfile.deploy` copies source and the reference bundle into an image. Its build runs a deep checksum check. `compose.deploy.yaml` has no source/model bind mounts. This is the same Dockerfile the EC2 bootstrap uses. An alternate experiment is not promoted automatically: deliberately restore/export the selected reference bundle before building.

```sh
cp .env.deploy.example .env.deploy
docker compose --env-file .env.deploy -f compose.deploy.yaml up --build -d
curl --fail http://127.0.0.1:8501/_stcore/health
docker compose --env-file .env.deploy -f compose.deploy.yaml logs --tail=100 app
# After browser inference verification:
docker compose --env-file .env.deploy -f compose.deploy.yaml down
```

Both Compose configurations bind only to localhost by default. The EC2 service directly runs the deployment image and exposes 8501 subject to the security group's allowed CIDR. Docker image execution still requires your validation: the Docker daemon was unavailable during this handoff. Compose parsing, shell syntax, bundle integrity and native Streamlit inference were checked; these do not substitute for an image build/run.

## 5. Final AWS deployment after credentials are available

**Current procedure (updated 11 November 2026).** Use the make targets; the subsections further down describe the original JSON-config route, which still works through `--config` but is no longer needed.

1. `make aws-deploy` once creates `.env.aws` from `.env.aws.example`. Paste `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and (Learner Lab) `AWS_SESSION_TOKEN`. Keep `AWS_DEFAULT_REGION=us-east-1` (Learner Lab allows us-east-1 and us-west-2 only).
2. `make aws-plan` checks the bundle and prints the stack, version name, upload size and estimated cost ($0.0832/h for `t3.large` plus about $0.004/h for the 40 GB disk, on-demand us-east-1 prices). Nothing changes in AWS.
3. `make aws-deploy` signs in, creates a private bucket `job-recommender-<account>-<region>` (unless `S3_BUCKET` is set), streams the allowlisted files to `<stack>/packages/<version>.tar.gz` with a metadata file in `<stack>/versions/`, creates the stack and waits for the health URL. A Learner Lab login is detected and the template uses `LabInstanceProfile` instead of creating an IAM role. If an earlier attempt left the stack in `ROLLBACK_COMPLETE`, it is deleted first; other half-finished states stop with the exact command to run.
4. Then the helper creates the CloudWatch log groups (below). If the account forbids it, this is skipped with a printed reason.

### Updates and rollback

`make aws-update` uploads a new versioned package and runs the install on the existing server through AWS Systems Manager (no CloudFormation update, no new server). On the server, `deploy/aws/release.sh`:

1. reuses the image if it is already on the server, else builds it (dependency layers are cached, so only the source and model layers change);
2. runs `python -m recommender.smoke` in the new image (loads the bundle, ranks a sample profile);
3. starts the new image next to the live app on `127.0.0.1:8502` and waits for its health URL;
4. switches the systemd service to the new image tag (`job-recommender:<version>`); if the switched app is not healthy within three minutes, it restores the previous service and version automatically.

The previous version stays stored in S3 (package and metadata), and the server keeps its last three images. `make aws-rollback` returns to the previous version through the same steps (fast, the image is already present) and then checks the health URL from your computer. `make aws-rollback VERSION=<name>` picks any stored version; `make aws-versions` lists them. Stacks created before versioning work too: their original package is registered as a version on the first update, rollback or versions call, and a failed first update restores the original service. The running version, commit and time are written to `artifacts/ec2-status.json` (with a history) and shown in the app sidebar and footer.

If Systems Manager is not reachable (agent offline or not allowed), update and rollback stop with a message; the fallback is `make aws-destroy` then `make aws-deploy` (short downtime). If the server is stopped (end of a Learner Lab session), update and rollback start it first and wait for Systems Manager.

### Logs and self-healing

- **Logs:** the CloudWatch agent ships container logs to `/job-recommender/<stack>/app` and the bootstrap log to `/job-recommender/<stack>/bootstrap`, kept for `LOG_RETENTION_DAYS` (default 14). `make aws-logs` reads them, or asks the server directly when CloudWatch is unavailable.
- **Container self-healing:** `job-recommender-health.timer` checks the health URL every minute and restarts the container after three failures in a row; systemd also restarts it if it exits.
- **Cost:** the Learner Lab caps spending at $50; check "Used $X of $50" on the lab page. `make aws-plan` prints the estimated hourly cost.
- `make aws-status` shows the stack and server state, the current address (it changes after a stop/start), the running version and a live health check.

### Access and the optional HTTPS route (off by default)

`ALLOWED_CIDR` (empty = your current public IP `/32`) is the address range allowed at deploy time on port 8501, and the app is plain HTTP. To add another network without redeploying, run `make aws-allow CIDR=<address>/32` (`CIDR=me` adds your current IP, for example after your IP changes); `make aws-revoke CIDR=<address>/32` removes it. For a presentation, connect to the room's network, open <https://checkip.amazonaws.com> to see its public address, run `make aws-allow CIDR=<that address>/32`, and revoke it afterwards. `CIDR=0.0.0.0/0` opens the app to everyone; use it only for the demo window. To show the app to people outside your IP, do not open `0.0.0.0/0` over HTTP; use one of these, which are documented options and not part of the template:

- **CloudFront in front of the instance:** create a CloudFront distribution with the instance's public DNS name as an HTTP origin on port 8501, viewer protocol "Redirect HTTP to HTTPS", caching disabled and WebSocket support (Streamlit needs it; CloudFront supports WebSockets by default). Restrict the security group to the CloudFront origin-facing managed prefix list (`com.amazonaws.global.cloudfront.origin-facing`). This gives an `https://<id>.cloudfront.net` address without owning a domain. Add authentication before sharing real profiles.
- **Application Load Balancer with an ACM certificate:** needs a domain name you control, two subnets in different availability zones and a target group on port 8501 with health check path `/_stcore/health`. Higher fixed cost (about $16/month for the load balancer).

Learner Lab may not allow CloudFront or ACM; check its "Service usage and other restrictions" page first.

### Runbook: app is down

1. `make aws-status`. If the server is `stopped`, start the lab and run `make aws-status` again (an update or rollback also starts it). Note the new address.
2. If health is "NOT ANSWERING": `make aws-logs` and look for a Python error or `Model bundle missing`.
3. If your IP changed, the page will not load even though health on the server is fine: update `ALLOWED_CIDR`.
4. If the latest release is the cause, `make aws-rollback`. If the server itself is broken (no Systems Manager, status checks failing in the EC2 console), reboot it from the EC2 console or run `make aws-destroy` then `make aws-deploy`.

### Runbook: bad release

1. If `make aws-update` failed, nothing needs undoing: the running version was kept (candidate unhealthy or smoke test failed) or restored (switch failed). The output says which.
2. If the release went live but behaves badly, `make aws-rollback`, then confirm in the browser and with `make aws-status`.
3. Fix the code, `make test`, commit, then `make aws-update` again. Old versions remain listed by `make aws-versions` until `make aws-destroy`.


No AWS resources have been created. The helper creates a **new** CloudFormation stack; it is not an in-place update system. Default execution builds a local plan only. `--apply` is the explicit cloud-changing command. Git push is independent of this deployment; EC2 receives an allowlisted private package, not a clone of your repository.

### Fill-in values

| File or setting | Placeholder/value to change | Meaning |
|---|---|---|
| `deploy/aws/deployment.local.json` | `region` | Your AWS region, for example `ap-southeast-1`; use the same region for the bucket and stack. |
| same | `bucket` | Existing private S3 bucket name, without `s3://`; create it in your account first. Block public access. |
| same | `allowed_cidr` | Your public IPv4 plus `/32`, e.g. `203.0.113.10/32` is a documentation example to replace. Controls access to app port 8501. |
| same | `profile` | Your configured AWS CLI/SDK profile name; blank uses the standard credential chain. |
| same | `stack_name` | Unique stack name; default `job-recommender`. Use a new name for a new deployment. |
| same | `instance_type` | Default `t3.large` (8 GB); supported alternatives `t3.medium`, `t3.xlarge`. Allow build/runtime memory and disk headroom. |
| `.env.deploy` | `IMAGE_TAG`, `DEPLOY_PORT`, `BIND_ADDRESS` | Local deployment-rehearsal image name, port and bind address. Defaults are safe for localhost rehearsal. These do not modify EC2 settings. |
| shell environment | `MODEL_BUNDLE` | Optional native app bundle override; not an AWS credential. |
| command examples | `PROJECT_PROFILE`, `AWS_REGION`, `STACK_NAME`, `INSTANCE_ID`, `BUCKET_NAME`, `ARTIFACT_KEY` | Replace uppercase example tokens with your actual values before running those commands. |
| rebuild examples | `/absolute/path/to/encoder/snapshot`, run names | Your local encoder path if needed and a new unused run directory. |

Do not replace `${AWS::Region}`, `${AWS::StackName}`, `${ArtifactBucket}`, `${ArtifactKey}`, `${ArtifactSha256}` or other `${...}` expressions inside `ec2-stack.json` manually: CloudFormation substitutes them. Generated instance IDs, IAM roles, VPC/subnet/security-group IDs, package SHA and S3 object key are supplied by the template/helper, not credentials to paste into source. Existing local path examples must be changed when using another computer.

### Authenticate and prepare

Install AWS CLI v2 separately. Use your organization's approved AWS login. If the account uses IAM Identity Center:

```sh
aws configure sso --profile PROJECT_PROFILE
aws sso login --profile PROJECT_PROFILE
aws sts get-caller-identity --profile PROJECT_PROFILE
```

If you already have a working profile, use it. Do not put access keys, secret keys or private keys into JSON, Dockerfiles or Git. The deploying identity needs scoped permissions for CloudFormation, EC2/VPC creation, IAM role creation/pass-role and S3 package access; SSM administration permissions are needed for logs. Arrange these with your account administrator. An AWS profile name is not a secret key.

```sh
python -m pip install -r requirements-deploy.txt
sh deploy/preflight.sh
cp deploy/aws/deployment.example.json deploy/aws/deployment.local.json
# Edit deployment.local.json with the values above, then:
python -m recommender.bundle check --deep
python deploy/aws/deploy.py --config deploy/aws/deployment.local.json
```

The last command verifies the bundle, creates `artifacts/ec2-deployment.tar.gz` and `artifacts/ec2-plan.json`, and prints the plan without contacting AWS. To only create the package, use `--package-only`. The package contains source and serving catalog/model files, excludes raw resumes and outcomes, and must be kept private. Review account, region, bucket, CIDR and expected charges before applying.

### Create and verify

First complete the deployment Docker rehearsal above. Then:

```sh
python deploy/aws/deploy.py --config deploy/aws/deployment.local.json --apply
```

The helper checks authentication, uploads an AES256-encrypted S3 package, then creates the stack. EC2 verifies its SHA-256, builds `Dockerfile.deploy`, and starts a systemd-managed container. The template provides an encrypted 40 GB volume, IMDSv2, SSM role and no SSH ingress. Package hash validation protects transfer integrity; model manifest validation protects bundle consistency.

Read `artifacts/ec2-status.json` for `InstanceId`, `AppUrl` and `HealthUrl`. CloudFormation completion can precede application bootstrap completion. Open the emitted health URL and the app from your allowed network; submit a profile and verify actual results. If bootstrap fails:

```sh
aws ssm start-session --target INSTANCE_ID --region AWS_REGION --profile PROJECT_PROFILE
# Inside the instance:
sudo tail -n 100 /var/log/job-recommender-bootstrap.log
sudo journalctl -u job-recommender --no-pager -n 100
sudo docker logs --tail=100 job-recommender
```

Session Manager may require its local CLI plugin. Internet access is needed on EC2 to install packages and build the image; inference uses the bundled offline encoder. The public IP can change after a stop/start. This template exposes restricted HTTP for a demonstration; a public service handling real profiles needs TLS, authentication and an appropriate data/privacy review before wider access.

### Stop costs and release strategy

For another release, use a new unique stack name, verify the new endpoint, then remove the old stack. This avoids claiming a user-data edit redeploys an existing instance. For cleanup:

```sh
aws cloudformation delete-stack --stack-name STACK_NAME --region AWS_REGION --profile PROJECT_PROFILE
aws cloudformation wait stack-delete-complete --stack-name STACK_NAME --region AWS_REGION --profile PROJECT_PROFILE
aws s3 rm s3://BUCKET_NAME/ARTIFACT_KEY --region AWS_REGION --profile PROJECT_PROFILE
```

Find the exact artifact key in the saved plan; delete only your deployment object. The existing bucket is not created or deleted by this stack. Versioned S3 buckets can retain previous object versions; review those separately. Confirm cleanup in AWS; stopping a container alone does not remove EC2/EBS/S3 charges.

## 6. Final Git handoff and project limits

```sh
python -m recommender.pipeline review
python -m recommender.bundle check --deep
git status --short
git diff --check
git diff --stat
```

Review the changes, commit and push yourself. No commit/push has been made. Keep code, seven notebooks, frozen dictionaries, aggregate results and docs. `.gitignore` excludes raw/clean data, embeddings, applicant-level results, bundles/runs/packages, local environment files, AWS details and secrets. Example environment/config files remain tracked. No test source files were added; runtime checks are explicit CLI preflights and the consistency review.

The local research-to-app pipeline is implemented. Remaining user steps are: run the Docker rehearsal (`make up`) with a working daemon, fill AWS details, authenticate, inspect the plan, apply, verify live inference and clean up resources when finished. Docker execution and AWS deployment are not claimed complete.

Research limitations remain: invitation-only positive labels; non-invited candidates are weak negatives, not confirmed bad matches; cleaning produces outcome-dependent snapshots; no temporal future-job validation; historical vacancies may be stale; the dictionary is small and pending human review; prior test inspection and post-exploration promotion policy limit confirmatory claims. Stronger strict/NDCG results from learned rerankers should remain visible even though M8 is retained on the primary criterion. This is a completed project prototype, not proof of hiring outcomes or a production readiness certification.

## 7. References and evidence

Local evidence: `results/final/selection.json`, `development_promotion.csv`, `metrics_grouped.csv`, `metrics_strict.csv`, `significance.csv`, `region_counts.csv`, `metrics_by_region.csv`, `run_manifest.json`, `artifacts/deployment/metadata.json` and `docs/final_review.json`. The model verdict provides more research interpretation.

Official operational references: [Docker Compose services and bind mounts](https://docs.docker.com/reference/compose-file/services/), [Streamlit Docker deployment and health endpoint](https://docs.streamlit.io/deploy/tutorials/docker), [AWS CLI SSO setup](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html).

## Desktop-copy local fix

The complete checked serving bundle has been restored into this Desktop copy at `artifacts/deployment/`. No Downloads path is embedded in the code. App and CLI defaults resolve from the installed project folder; relative run, input, output and bundle options resolve against the selected project root. AWS relative configuration paths resolve against the source project. Absolute paths remain supported. When moving the project, copy the complete private bundle separately because Git/source ZIPs intentionally exclude it.

## English input and output

The UI, region labels, vacancy titles/descriptions/skills and downloaded CSV are in English. Profiles are translated English → Russian before calling the existing recommender; returned catalog text is translated Russian → English. Region selections retain their original catalog keys. Listed salaries stay in RUB, with no currency conversion. Translation is a serving adapter; it does not modify M8 weights, the original encoder, catalog, experiment results or selection decision.

The adapter uses two pinned Helsinki-NLP OPUS-MT models, loaded on CPU through Transformers. Model cards: [English–Russian](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru) and [Russian–English](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en). Revisions and hashes are recorded in `artifacts/translation/provenance.json` and `manifest.json`. Model READMEs are retained with their assets for attribution/license information.

The Desktop project already has the translation assets after this change. For a new clone or missing assets, install the updated requirements, then prepare them once with Internet access:

```sh
python -m pip install -r requirements-app.txt
python -m recommender.translation prepare
python -m recommender.translation check --deep
streamlit run app.py
```

The app uses local-only model loading, so profile text is never sent to a translation API. The preparation command downloads model files only. User profiles are not cached or written to disk. Long fields are token-chunked during translation instead of being silently truncated there; M8 still retains its original 400-character semantic-input policy.

`TRANSLATION_BUNDLE` optionally selects another complete translation folder. Like `MODEL_BUNDLE`, relative paths resolve from the source project. Deployment Docker copies the translation bundle and checks its manifest at build time. The AWS helper validates and includes it in the private deployment package. No API keys or new AWS credentials are required for translation.

Translation adds CPU latency, disk and memory usage; the first recommendation loads both translation models. Keep the recommended 8 GB EC2 instance and verify container memory under actual use. Translation may mistranslate names, skills and specialist terminology. English examples have smoke validation, but the published Recall/NDCG results evaluate original Russian profiles, not this new English adapter; a bilingual relevance/translation evaluation would be needed to claim equal English retrieval quality.

## Structured profile form and skill matches

The app now provides job-title, experience and education dropdowns; responsibilities and skills multi-selects; and the existing English region selector. Choose Other plus a custom job title if your occupation is absent. One optional notes field captures additional details. Unknown experience/education are omitted rather than invented. Skills and responsibilities become a consistent profile narrative; selected skills also retain original dictionary canonical aliases so English translation does not change their identity. These inputs guide retrieval, not a new trained qualification/experience filter.

Each vacancy card shows `1 matched skill`, `2 matched skills` or `0 matched skills`, followed by the matching English skill names. The exported CSV includes Matched skill count and Matched skills. Counts intersect your selected canonical skill IDs with positive entries in the exported vacancy skill matrix. Aliases of the same skill count once; negated/non-required evidence and soft traits are excluded by the existing extraction contract. These are exact dictionary-ID overlaps, not literal English string matches or a probability. Unselected skills inferred from optional prose do not inflate the displayed count. Zero can mean missing/limited evidence. M8 ranking remains unchanged; this is an input-format improvement and result explanation, not M9 promotion. Implementation: `recommender/profile_form.py`.
