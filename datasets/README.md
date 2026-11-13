# Datasets

| File | Used by | Tracked in Git |
|---|---|---|
| `HSE_dataset(D4).csv` | Main dataset: notebooks 01–04, `recommender.pipeline`, skill extraction | No (about 1 GB) |
| `ai_job_dataset.csv` | Supplementary salary notebook only (`notebooks/archive/`) | No (licence not confirmed) |
| `ai_job_salary_dataset_10k.csv` | Same notebook, cross-check only | No (source not confirmed) |

## Main dataset (D4)

HSE resume and vacancy hackathon data (Russian labour market), about 150,000 applications:
<https://www.kaggle.com/datasets/darysha/hse-hackathon>

1. Download it and save it as `datasets/HSE_dataset(D4).csv`.
2. The file is separated by `|`, not commas.
3. If you keep another file name, the cleaning code uses the first `.csv` in this folder with `hse` in its name.
4. Gender, age and birthday are never loaded, so no model can use them.

## Supplementary salary files (not used by the recommender)

The salary files are supporting analysis from the proposal stage. Following the professor feedback, salary prediction is not a deliverable, and no part of the recommender, the pipeline or the app reads them. They are not distributed in this repository because their licences could not be confirmed. The archived notebook keeps its saved outputs, so its results can be read without the files.

To re-run `notebooks/archive/00_salary_analysis_supplementary.ipynb`, place both files in this folder:

- `ai_job_dataset.csv` (15,000 rows, 19 columns: `job_id`, `job_title`, `salary_usd`, …): matches the Kaggle dataset "Global AI Job Market & Salary Trends 2025", <https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025>. Check its licence on that page before redistributing.
- `ai_job_salary_dataset_10k.csv` (10,000 rows: `id`, `job_role`, `country`, `experience_level`, `salary_usd`, `skills`, `company_type`): obtained during the proposal stage; the original source is not recorded.
