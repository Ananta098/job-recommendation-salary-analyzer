# Datasets

| File | Used by | Tracked in Git |
|---|---|---|
| `HSE_dataset(D4).csv` | Main dataset: notebooks 01–04, `recommender.pipeline`, skill extraction | No (about 1 GB, private) |
| `ai_job_dataset.csv` | Supplementary salary notebook (`notebooks/archive/00_salary_analysis_supplementary.ipynb`) | Yes |
| `ai_job_salary_dataset_10k.csv` | Same notebook, used as a cross-check | Yes |

## Main dataset (D4)

HSE resume and vacancy hackathon data (Russian labour market), about 150,000 applications:
<https://www.kaggle.com/datasets/darysha/hse-hackathon>

1. Download it and save it as `datasets/HSE_dataset(D4).csv`.
2. The file is separated by `|`, not commas.
3. If you keep another file name, the cleaning code uses the first `.csv` in this folder with `hse` in its name.
4. Gender, age and birthday are never loaded, so no model can use them.

The salary files are supporting analysis from the proposal stage. Following the professor feedback, salary prediction is not a deliverable and the recommender does not read them.
