# Final pipeline review

Reviewed 3 November 2026. **The local project/prototype pipeline is complete and ready for Git review.** M8 remains the selected serving model. No commit or push was performed.

## Verified flow

1. Raw D4 cleaning: executed independently from the new shared cleaning module; the resulting CSV is byte-identical to the evaluated input.
2. Prepared-input reuse: verified all recorded input hashes and the raw D4 hash before reuse. Encoding/extraction already completed for those exact inputs was not repeated.
3. Fresh isolated comparison: the new pipeline command fitted/evaluated M8, both M9 variants, Logistic Regression, Random Forest and XGBoost, and evaluated M5 as comparator. Applicant splits remained disjoint.
4. Selection and export: M8 passed the documented retention/promotion policy and a new serving bundle was exported.
5. Analysis: the pipeline generated paired significance, bootstrap intervals, regional sample counts and regional performance in the same run directory.
6. Final review: bundle integrity, selected-model agreement, regional totals, applicant/aggregate consistency and source syntax passed.
7. Streamlit: the refactored imports, selected bundle, recommendation call and form submission returned ten recommendations without an uncaught exception.
8. Reports: notebooks 02–07 executed successfully as report views. Notebook 01 retains the existing optional raw-data EDA.
9. Deployment files: CloudFormation lint and Compose configuration checks passed. The deployment package is generated from the current source/model allowlist.

The full rebuild route also contains pinned encoder resolution, fresh embedding generation and skill extraction. The verification run used checked existing embeddings/sidecars to avoid repeating completed work; it is not described as a second full re-encoding of the entire catalog. Original encoder/cache agreement and inference-parity checks are retained in the review record.

## Redundancy removed

- Removed the standalone earlier M9 runner and regional-only result copy; both M9 variants now have one authoritative comparison in `results/final/`.
- Extracted the shared split, normalization/evidence helpers and metrics into one module.
- Moved cleaning out of notebook-only code into one callable implementation.
- Converted overlapping notebook training procedures to report views.
- Kept one paired significance procedure; confidence intervals provide complementary uncertainty, without a second Wilcoxon test.
- Separated historical baseline evidence from current results and removed subset copies of the same metrics.
- Removed superseded documents, duplicate ranking plots, duplicate SVG copies and unused initial extraction sidecars. Distinct dictionary-review evidence and baseline linguistic diagnostics remain.
- Consolidated all run/deployment instructions into `docs/PROJECT_HANDBOOK.md`.

## External limits and scientific scope

- **Docker image execution is not verified:** the local Docker daemon remains unavailable. Start it and run the documented build/health commands before relying on the container.
- **AWS is intentionally a template:** fill in your deployment details and authenticate when you are ready. No cloud resources or live URL were created.
- For public use with real resumes, add TLS and authentication; the template provides a restricted HTTP demonstration endpoint.
- Invitation-only evaluation, weak negatives, outcome-dependent snapshots, prior test-set inspection and pending human dictionary review remain explicit. No temporal/future-job or production-quality guarantee is claimed.

These are external execution/research limitations, not missing hidden stages in the local code. The machine-readable evidence is in `docs/final_review.json`; the model interpretation is in `docs/final_model_verdict.md`.

## Docker and bundle handoff update

Default/relative bundle paths now resolve from the project root. Lightweight preflight reports the resolved missing path; the serving loader retains full checksum validation. Separate `Dockerfile.local`/`compose.local.yaml` mount source/model for testing; `Dockerfile.deploy`/`compose.deploy.yaml` bake a deeply checked bundle into the image. EC2 packaging/bootstrap use the deployment Dockerfile. Shared shell scripts avoid duplicate setup logic. The handbook lists every user configuration placeholder. Docker execution remains pending a working daemon.

## English interface verification

English profile input now passes through local pinned OPUS-MT English→Russian translation before the unchanged M8 scorer. Vacancy fields and CSV are translated Russian→English; region display labels keep original matching keys. Both translation bundles were prepared and SHA-256 verified. Streamlit AppTest from `/tmp`, with Hugging Face offline flags enabled, submitted an English accounting profile, selected an English-labelled Moscow region and returned ten English result headings. English CSV checks preserved vacancy identifiers and contained no Cyrillic text for the sample; blank/already-English output handling passed. Original pipeline review, source syntax, Compose configurations and Git whitespace checks passed. Docker runtime remains unverified because the daemon is unavailable. These are smoke checks, not a bilingual retrieval benchmark.
